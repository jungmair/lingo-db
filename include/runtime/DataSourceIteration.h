#ifndef RUNTIME_DATASOURCEITERATION_H
#define RUNTIME_DATASOURCEITERATION_H
#include "RecordBatchInfo.h"
#include "runtime/ExecutionContext.h"
#include "runtime/helpers.h"
namespace runtime {
class DataSourceIterator {
   public:
   virtual std::shared_ptr<arrow::RecordBatch> getNext() = 0;
   virtual ~DataSourceIterator() {}
};
class DataSource {
   public:
   virtual size_t getColumnId(std::string member) = 0;
   virtual std::shared_ptr<DataSourceIterator> getIterator() = 0;
   virtual ~DataSource() {}
   static DataSource* get(ExecutionContext* executionContext, runtime::VarLen32 description);
};
class DataSourceIteration {
   std::shared_ptr<arrow::RecordBatch> currChunk;
   DataSource* dataSource;
   std::shared_ptr<DataSourceIterator> iterator;
   std::vector<size_t> colIds;

   public:
   DataSourceIteration(DataSource* dataSource, const std::vector<size_t>& colIds);

   static DataSourceIteration* init(DataSource* dataSource, runtime::VarLen32 members);
   bool isValid();
   void next();
   void access(RecordBatchInfo* info);
   static void end(DataSourceIteration*);
   void iterate(void (*forEachChunk)(RecordBatchInfo*,void*),void*);
};
} // end namespace runtime
#endif // RUNTIME_DATASOURCEITERATION_H
