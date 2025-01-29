#include "lingodb/runtime/Buffer.h"
#include "lingodb/utility/Tracer.h"
#include <iostream>
namespace {
static utility::Tracer::Event iterateEvent("FlexibleBuffer", "iterateParallel");
static utility::Tracer::Event bufferIteratorEvent("BufferIterator", "iterate");

class FlexibleBufferIteratorTask : public lingodb::scheduler::Task {
   std::vector<lingodb::runtime::Buffer>& buffers;
   size_t typeSize;
   const std::function<void(lingodb::runtime::Buffer)> cb;
   std::mutex mutex;
   std::vector<size_t> bufferSteps;
   size_t startIndex{0};
   size_t bufferIndex{0};
   size_t splitSize{20000};

   public:
   FlexibleBufferIteratorTask(std::vector<lingodb::runtime::Buffer>& buffers, size_t typeSize, const std::function<void(lingodb::runtime::Buffer)> cb) : buffers(buffers), typeSize(typeSize), cb(cb) {
      for (size_t i = 0; i < buffers.size(); i ++) {
         auto step = (buffers[i].numElements + splitSize - 1) / splitSize;
         if (i == 0) {
            bufferSteps.push_back(step);
         } else {
            bufferSteps.push_back(step+bufferSteps[i-1]);
         }
      }
   }
   void run() override {
      size_t bufferOffset = 0;
      {
         const std::lock_guard<std::mutex> lock(mutex);
         if (bufferIndex >= bufferSteps.size()) {
            workExhausted.store(true);
            return;
         }
         if (startIndex >= bufferSteps[bufferIndex]) {
            bufferIndex++;
            if (bufferIndex >= bufferSteps.size()) {
               workExhausted.store(true);
               return;
            }
         } else if (bufferIndex > 0) {
            bufferOffset = startIndex - bufferSteps[bufferIndex-1];
         }
         startIndex++;
      }
      auto& buffer = buffers[bufferIndex];
      utility::Tracer::Trace trace(iterateEvent);

      size_t begin = splitSize*bufferOffset;
      size_t end = std::min(begin + splitSize, buffer.numElements);
      size_t len = end - begin;
      auto buf = lingodb::runtime::Buffer{len, buffer.ptr + begin * std::max(1ul, typeSize)};
      cb(buf);

      trace.stop();
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
