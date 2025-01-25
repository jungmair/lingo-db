#ifndef LINGODB_RUNTIME_BUFFER_H
#define LINGODB_RUNTIME_BUFFER_H
#include <stddef.h>

#include "ExecutionContext.h"
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace lingodb::runtime {
/*
 * Buffer: continuous memory area, which is directly accessed by generated code
 */
struct BufferIterator;
struct Buffer {
   size_t numElements;
   uint8_t* ptr;
   static Buffer createZeroed(runtime::ExecutionContext* executionContext, size_t bytes);
   static void iterate(bool parallel, Buffer, size_t typeSize, void (*forEachChunk)(Buffer, size_t, size_t, void*), void* contextPtr);
};
struct BufferIterator {
   virtual bool isValid() = 0;
   virtual void next() = 0;

   virtual Buffer getCurrentBuffer() = 0;
   virtual void iterateEfficient(bool parallel, void (*forEachChunk)(Buffer, void*), void*) = 0;
   static bool isIteratorValid(BufferIterator* iterator);
   static void iteratorNext(BufferIterator* iterator);

   static Buffer iteratorGetCurrentBuffer(BufferIterator* iterator);
   static void destroy(BufferIterator* iterator);
   static void iterate(BufferIterator* iterator, bool parallel, void (*forEachChunk)(Buffer, void*), void*);
   virtual ~BufferIterator() {}
};
class FlexibleBuffer {
   size_t totalLen;
   size_t currCapacity;
   std::vector<Buffer> buffers;
   size_t typeSize;

   void nextBuffer() {
      size_t nextCapacity = currCapacity * 2;
      buffers.push_back(Buffer{0, (uint8_t*) malloc(nextCapacity * typeSize)});
      currCapacity = nextCapacity;
   }

   public:
   FlexibleBuffer(size_t initialCapacity, size_t typeSize) : totalLen(0), currCapacity(initialCapacity), typeSize(typeSize) {
      buffers.push_back(Buffer{0, (uint8_t*) malloc(initialCapacity * typeSize)});
   }
   uint8_t* insert() {
      if (buffers.back().numElements == currCapacity) {
         nextBuffer();
      }
      totalLen++;
      auto* res = &buffers.back().ptr[typeSize * (buffers.back().numElements)];
      buffers.back().numElements++;
      return res;
   }
   void iterateBuffersParallel(const std::function<void(Buffer)>& fn);
   template <class Fn>
   void iterate(const Fn& fn) {
      for (auto buffer : buffers) {
         for (size_t i = 0; i < buffer.numElements; i++) {
            fn(&buffer.ptr[i * typeSize]);
         }
      }
   }
   template <class Fn>
   void iterateParallel(const Fn& fn) {
      iterateBuffersParallel([&](auto buffer) {
         for (size_t i = 0; i < buffer.numElements; i++) {
            fn(&buffer.ptr[i * typeSize]);
         }
      });
   }
   const std::vector<Buffer>& getBuffers() {
      return buffers;
   }
   BufferIterator* createIterator();
   size_t getTypeSize() const {
      return typeSize;
   }
   size_t getLen() const;
   void merge(FlexibleBuffer& other) {
      buffers.insert(buffers.begin(), other.buffers.begin(), other.buffers.end());
      other.buffers.clear();
      totalLen += other.totalLen;
      other.totalLen = 0;
      other.currCapacity = 0;
   }
   ~FlexibleBuffer() {
      for (auto buf : buffers) {
         free(buf.ptr);
      }
   }
};

// TODO: maybe need a Task class with template T
// std::vector<T>& buffers;
// const std::function<void(T)> cb;
class DispatchBufferTask : public lingodb::scheduler::Task {
   std::vector<Buffer>& buffers;
   const std::function<void(Buffer)> cb;
   std::atomic<size_t> startIndex{0};

   public:
   DispatchBufferTask(std::vector<Buffer>& buffers, const std::function<void(Buffer)> cb) : buffers(buffers), cb(cb) {}
   void run() override;
};

class SplitBufferTask : public lingodb::scheduler::Task {
   Buffer& buffer;
   size_t bufferLen;
   void* contextPtr;
   const std::function<void(lingodb::runtime::Buffer, size_t, size_t, void*)> cb;
   size_t splitSize{20000};
   std::atomic<size_t> startIndex{0};

   public:
   SplitBufferTask(Buffer& buffer, size_t typeSize, void* contextPtr, const std::function<void(lingodb::runtime::Buffer, size_t, size_t, void*)> cb) : buffer(buffer), bufferLen(buffer.numElements / typeSize), contextPtr(contextPtr), cb(cb) {}
   void run() override;
};

} // namespace lingodb::runtime

#endif //LINGODB_RUNTIME_BUFFER_H
