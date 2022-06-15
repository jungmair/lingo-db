//
// Created by michael on 09.06.22.
//

#ifndef DB_DIALECTS_CRANELIFTCONVERSIONS_H
#define DB_DIALECTS_CRANELIFTCONVERSIONS_H
namespace mlir::cranelift{
void registerCraneliftConversionPasses();
std::unique_ptr<Pass> createLowerToCraneliftPass();

}

#endif //DB_DIALECTS_CRANELIFTCONVERSIONS_H
