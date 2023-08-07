/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may
   not use this file except in compliance with the License. You may obtain
   a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
   License for the specific language governing permissions and limitations
   under the License.


*/
//===----------------------------------------------------------------------===//
//
// This is the top level file for the plugin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PLUGIN_DIALECT_TD
#define PLUGIN_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Plugin Dialect Definition.
//===----------------------------------------------------------------------===//

// Provide a definition of the 'Plugin' dialect in the ODS framework so that we
// can define our operations.
def Plugin_Dialect : Dialect {
  let name = "Plugin";
  let summary = "The Plugin Dialect.";
  let description = [{}];
  let cppNamespace = "::mlir::Plugin";

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

// Base class for Plugin dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Plugin_Op<string mnemonic, list<Trait> traits = []> :
    Op<Plugin_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// PluginDialect enum definitions
//===----------------------------------------------------------------------===//

def IComparisonLT    : I32EnumAttrCase<"lt", 0>;
def IComparisonLE    : I32EnumAttrCase<"le", 1>;
def IComparisonGT    : I32EnumAttrCase<"gt", 2>;
def IComparisonGE    : I32EnumAttrCase<"ge", 3>;
def IComparisonLTGT  : I32EnumAttrCase<"ltgt", 4>;
def IComparisonEQ    : I32EnumAttrCase<"eq", 5>;
def IComparisonNE    : I32EnumAttrCase<"ne", 6>;
def IComparisonUNDEF : I32EnumAttrCase<"UNDEF", 7>;
def IComparisonCode  : I32EnumAttr<
   "IComparisonCode", "plugin comparison code",
   [IComparisonLT, IComparisonLE, IComparisonGT, IComparisonGE, 
    IComparisonLTGT, IComparisonEQ, IComparisonNE, IComparisonUNDEF]>{
  let cppNamespace = "::mlir::Plugin";
}

def IDefineCodeMemRef    : I32EnumAttrCase<"MemRef", 0>;
def IDefineCodeIntCST    : I32EnumAttrCase<"IntCST", 1>;
def IDefineCodeSSA       : I32EnumAttrCase<"SSA", 2>;
def IDefineCodeLIST      : I32EnumAttrCase<"LIST", 3>;
def IDefineCodeStrCST    : I32EnumAttrCase<"StrCST", 4>;
def IDefineCodeArrayRef  : I32EnumAttrCase<"ArrayRef", 5>;
def IDefineCodeDecl      : I32EnumAttrCase<"Decl", 6>;
def IDefineCodeFieldDecl : I32EnumAttrCase<"FieldDecl", 7>;
def IDefineCodeAddrExp   : I32EnumAttrCase<"AddrExp", 8>;
def IDefineCodeConstructor  : I32EnumAttrCase<"Constructor", 9>;
def IDefineCodeVec  : I32EnumAttrCase<"Vec", 10>;
def IDefineCodeBLOCK  : I32EnumAttrCase<"BLOCK", 11>;
def IDefineCodeCOMPONENT  : I32EnumAttrCase<"COMPONENT", 12>;
def IDefineCodeTYPEDECL   : I32EnumAttrCase<"TYPEDECL", 13>;
def IDefineCodeUNDEF     : I32EnumAttrCase<"UNDEF", 14>;
def IDefineCode  : I32EnumAttr<
   "IDefineCode", "plugin define code",
   [IDefineCodeMemRef, IDefineCodeIntCST, IDefineCodeSSA, IDefineCodeLIST,
   IDefineCodeStrCST, IDefineCodeArrayRef, IDefineCodeDecl, IDefineCodeFieldDecl, IDefineCodeAddrExp,
   IDefineCodeConstructor, IDefineCodeVec, IDefineCodeBLOCK, IDefineCodeCOMPONENT, IDefineCodeTYPEDECL,
   IDefineCodeUNDEF]>{
  let cppNamespace = "::mlir::Plugin";
}

def IExprCodePlus    : I32EnumAttrCase<"Plus", 0>;
def IExprCodeMinus   : I32EnumAttrCase<"Minus", 1>;
def IExprCodeMult    : I32EnumAttrCase<"Mult", 2>;
def IExprCodePtrPlus : I32EnumAttrCase<"PtrPlus", 3>;
def IExprCodeMin     : I32EnumAttrCase<"Min", 4>;
def IExprCodeMax     : I32EnumAttrCase<"Max", 5>;
def IExprCodeBitIOR  : I32EnumAttrCase<"BitIOR", 6>;
def IExprCodeBitXOR  : I32EnumAttrCase<"BitXOR", 7>;
def IExprCodeBitAND  : I32EnumAttrCase<"BitAND", 8>;
def IExprCodeLshift  : I32EnumAttrCase<"Lshift", 9>;
def IExprCodeRshift  : I32EnumAttrCase<"Rshift", 10>;
def IExprCodeNop     : I32EnumAttrCase<"Nop", 11>;
def IExprCodeUNDEF   : I32EnumAttrCase<"UNDEF", 12>;
def IExprCode  : I32EnumAttr<
   "IExprCode", "plugin expr code",
   [IExprCodePlus, IExprCodeMinus, IExprCodeMult,
    IExprCodePtrPlus, IExprCodeMin, IExprCodeMax,
    IExprCodeBitIOR, IExprCodeBitXOR, IExprCodeBitAND,
    IExprCodeLshift, IExprCodeRshift, IExprCodeNop,
    IExprCodeUNDEF]>{
  let cppNamespace = "::mlir::Plugin";
}

#endif // PLUGIN_DIALECT_TD