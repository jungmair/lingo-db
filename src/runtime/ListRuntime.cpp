#include "lingodb/runtime/ListRuntime.h"
using namespace lingodb::runtime;

List* List::create(size_t sizeOfType) {
   auto* list = new List(sizeOfType);
   getCurrentExecutionContext()->registerState({list, [](void* ptr) { delete reinterpret_cast<List*>(ptr); }});
   return list;
}
uint8_t* List::append() {
   if ((len + 1) * sizeOfType > values.size()) {
      values.resize(values.size() * 2);
   }
   auto* res = values.data() + len * sizeOfType;
   len++;
   return res;
}

Buffer List::getBuffer() {
   return Buffer(len * sizeOfType, values.data());
}
List* List::fromBuffer(size_t sizeOfType, Buffer buffer) {
   auto* res = new List(sizeOfType);
   res->len = buffer.numElements / sizeOfType;
   res->values.resize(buffer.numElements);
   memcpy(res->values.data(), buffer.ptr, buffer.numElements);
   return res;
}
uint8_t* List::at(size_t pos) {
   if (pos >= len) {
      throw std::runtime_error("out of bounds");
   }
   return values.data() + pos * sizeOfType;
}

size_t List::size() {
   return len;
}
void List::sort(bool (*isLess)(uint8_t*, uint8_t*)) {
   //vector is only used as byte storage, use type sizes
   if (len < 2) return; // No need to sort if there are less than 2 elements
   std::vector<uint8_t*> pointers(len);
   for (size_t i = 0; i < len; i++) {
      pointers[i] = values.data() + i * sizeOfType;
   }
   std::sort(pointers.begin(), pointers.end(), isLess);
   std::vector<uint8_t> sortedValues(len * sizeOfType);
   for (size_t i = 0; i < len; i++) {
      memcpy(sortedValues.data() + i * sizeOfType, pointers[i], sizeOfType);
   }
   values = std::move(sortedValues);
}

