#include "lingodb/runtime/Buffer.h"
#include "lingodb/utility/Tracer.h"
#include <iostream>
#include <mutex>
#include <boost/chrono/thread_clock.hpp>
namespace {
static utility::Tracer::Event iterateEvent("FlexibleBuffer", "iterateParallel");
static utility::Tracer::Event bufferIteratorEvent("BufferIterator", "iterate");

// TODO RENAME
class WorkerLocalState {
   public:
   std::mutex mutex;
   bool hasMore{false};
   int unitId{0};
   size_t unitAmount;
   size_t bufferId;
   // workerId steal task from
   size_t stealWorkerId{std::numeric_limits<size_t>::max()};

   int fetchAndNext() {
      size_t cur;
      {
         std::lock_guard<std::mutex> stateLock(this->mutex);
         cur = unitId;
         unitId++;
         hasMore = unitId < unitAmount;
      }
      if (cur >= unitAmount) {
         return -1;
      }
      return cur;
   }
};

class FlexibleBufferIteratorTask : public lingodb::scheduler::Task {
   std::vector<lingodb::runtime::Buffer>& buffers;
   size_t typeSize;
   const std::function<void(lingodb::runtime::Buffer)> cb;
   std::atomic<size_t> startIndex{0};
   size_t splitSize{20000};
   std::vector<std::unique_ptr<WorkerLocalState>> workerLocalStates;

   public:
   FlexibleBufferIteratorTask(std::vector<lingodb::runtime::Buffer>& buffers, size_t typeSize, const std::function<void(lingodb::runtime::Buffer)> cb) : buffers(buffers), typeSize(typeSize), cb(cb) {
      for (size_t i = 0; i < lingodb::scheduler::getNumWorkers(); i++) {
         workerLocalStates.emplace_back(std::make_unique<WorkerLocalState>());
      }
   }
   void unitRun(size_t bufferId, int unitId) {
      auto& buffer = buffers[bufferId];
      if (unitId < 0) {
         return;
      }
      size_t begin = splitSize*unitId;
      size_t len = std::min(begin + splitSize, buffer.numElements) - begin;
      auto buf = lingodb::runtime::Buffer{len, buffer.ptr + begin * std::max(1ul, typeSize)};
      cb(buf);
   }

   void run() override {
      auto state = workerLocalStates[lingodb::scheduler::currentWorkerId()].get();
      if (state->hasMore) {
         unitRun(state->bufferId, state->fetchAndNext());
         return;
      }

      // TODO TRACE
      if (startIndex < buffers.size()) {
         size_t localStartIndex = startIndex.fetch_add(1);
         if (localStartIndex < buffers.size()) {
            auto& buffer = buffers[localStartIndex];
            if (buffer.numElements < splitSize) {
               cb(buffer);
            } else {
               auto unitAmount = (buffer.numElements + splitSize - 1) / splitSize;
               {
                  std::lock_guard resetLock(state->mutex);
                  state->hasMore = true;
                  state->unitId = 1;
                  state->bufferId = localStartIndex;
                  state->unitAmount = unitAmount;
               }
               unitRun(state->bufferId, 0);
            }
            return;
         }
      }

      if (state->stealWorkerId != std::numeric_limits<size_t>::max()) {
         auto other = workerLocalStates[state->stealWorkerId].get();
         if (other->hasMore) {
            unitRun(other->bufferId, other->fetchAndNext());
            return;
         }
         state->stealWorkerId = std::numeric_limits<size_t>::max();
      }

      for (size_t i = 1; i < workerLocalStates.size(); i ++) {
         auto idx = lingodb::scheduler::currentWorkerId() + i;
         if (idx >= workerLocalStates.size()) {
            idx -= workerLocalStates.size();
         }
         auto other = workerLocalStates[idx].get();
         if (other->hasMore) {
            // only current worker can modify its onw stealWorkerId. no need to lock
            state->stealWorkerId = idx;
            unitRun(other->bufferId, other->fetchAndNext());
            return;
         }
      }

      workExhausted.store(true);
   }
};

class BufferIteratorTask : public lingodb::scheduler::Task {
   lingodb::runtime::Buffer& buffer;
   size_t bufferLen;
   void* contextPtr;
   const std::function<void(lingodb::runtime::Buffer, size_t, size_t, void*)> cb;
   size_t splitSize{20000};
   std::atomic<size_t> startIndex{0};

   public:
   BufferIteratorTask(lingodb::runtime::Buffer& buffer, size_t typeSize, void* contextPtr, const std::function<void(lingodb::runtime::Buffer, size_t, size_t, void*)> cb) : buffer(buffer), bufferLen(buffer.numElements / typeSize), contextPtr(contextPtr), cb(cb) {}
   void run() override {
      size_t localStartIndex = startIndex.fetch_add(1);
      if (localStartIndex*splitSize >= bufferLen) {
         workExhausted.store(true);
         return;
      }
      auto begin = localStartIndex*splitSize;
      auto end = (localStartIndex+1)*splitSize;
      if (end > bufferLen) {
         end = bufferLen;
      }
      utility::Tracer::Trace trace(iterateEvent);
      cb(buffer, begin, end, contextPtr);
      trace.stop();
   }

};

} // end namespace

bool lingodb::runtime::BufferIterator::isIteratorValid(lingodb::runtime::BufferIterator* iterator) {
   return iterator->isValid();
}
void lingodb::runtime::BufferIterator::iteratorNext(lingodb::runtime::BufferIterator* iterator) {
   iterator->next();
}
lingodb::runtime::Buffer lingodb::runtime::BufferIterator::iteratorGetCurrentBuffer(lingodb::runtime::BufferIterator* iterator) {
   return iterator->getCurrentBuffer();
}
void lingodb::runtime::BufferIterator::destroy(lingodb::runtime::BufferIterator* iterator) {
   delete iterator;
}
void lingodb::runtime::FlexibleBuffer::iterateBuffersParallel(const std::function<void(Buffer)>& fn) {
   lingodb::scheduler::awaitChildTask(std::make_unique<FlexibleBufferIteratorTask>(buffers, typeSize, fn));
}
class FlexibleBufferIterator : public lingodb::runtime::BufferIterator {
   lingodb::runtime::FlexibleBuffer& flexibleBuffer;
   size_t currBuffer;

   public:
   FlexibleBufferIterator(lingodb::runtime::FlexibleBuffer& flexibleBuffer) : flexibleBuffer(flexibleBuffer), currBuffer(0) {}
   bool isValid() override {
      return currBuffer < flexibleBuffer.getBuffers().size();
   }
   void next() override {
      currBuffer++;
   }
   lingodb::runtime::Buffer getCurrentBuffer() override {
      lingodb::runtime::Buffer orig = flexibleBuffer.getBuffers().at(currBuffer);
      return lingodb::runtime::Buffer{orig.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), orig.ptr};
   }
   void iterateEfficient(bool parallel, void (*forEachChunk)(lingodb::runtime::Buffer, void*), void* contextPtr) override {
      if (parallel) {
         flexibleBuffer.iterateBuffersParallel([&](lingodb::runtime::Buffer buffer) {
            buffer = lingodb::runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            forEachChunk(buffer, contextPtr);
         });
      } else {
         for (auto buffer : flexibleBuffer.getBuffers()) {
            buffer = lingodb::runtime::Buffer{buffer.numElements * std::max(1ul, flexibleBuffer.getTypeSize()), buffer.ptr};
            forEachChunk(buffer, contextPtr);
         }
      }
   }
};

lingodb::runtime::BufferIterator* lingodb::runtime::FlexibleBuffer::createIterator() {
   return new FlexibleBufferIterator(*this);
}
size_t lingodb::runtime::FlexibleBuffer::getLen() const {
   return totalLen;
}

void lingodb::runtime::BufferIterator::iterate(lingodb::runtime::BufferIterator* iterator, bool parallel, void (*forEachChunk)(lingodb::runtime::Buffer, void*), void* contextPtr) {
   utility::Tracer::Trace trace(bufferIteratorEvent);
   iterator->iterateEfficient(parallel, forEachChunk, contextPtr);
}

void lingodb::runtime::Buffer::iterate(bool parallel, lingodb::runtime::Buffer buffer, size_t typeSize, void (*forEachChunk)(lingodb::runtime::Buffer, size_t, size_t, void*), void* contextPtr) {
   size_t len = buffer.numElements / typeSize;

   auto range = tbb::blocked_range<size_t>(0, len);
   if (parallel) {
      // TODO: this is never triggered. parallel is set to false for window function
      lingodb::scheduler::awaitChildTask(std::make_unique<BufferIteratorTask>(buffer, typeSize, contextPtr, forEachChunk));
   } else {
      forEachChunk(buffer, 0, buffer.numElements / typeSize, contextPtr);
   }
}
