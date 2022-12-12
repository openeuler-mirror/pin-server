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

#ifndef PLUGIN_OPS_TD
#define PLUGIN_OPS_TD

include "PluginDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def FunctionOp : Plugin_Op<"function", [NoSideEffect]> {
  let summary = "function with a region";
  let description = [{
    TODO.
  }];

  let arguments = (ins UI64Attr:$id,
                       StrAttr:$funcName,
                       OptionalAttr<BoolAttr>:$declaredInline);
  let regions = (region AnyRegion:$bodyRegion);

  // Add custom build methods for the operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<Op>(...)`.
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$funcName, "bool":$declaredInline)>
  ];

  let extraClassDeclaration = [{
    std::vector<LoopOp> GetAllLoops();
    LoopOp AllocateNewLoop();
  }];
}

def LocalDeclOp : Plugin_Op<"declaration", [NoSideEffect]> {
  let summary = "local variable declaration";
  let description = [{TODO}];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<StrAttr>:$symName,
                       OptionalAttr<I64Attr>:$typeID,
                       OptionalAttr<UI64Attr>:$typeWidth);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$symName,
                  "int64_t":$typeID, "uint64_t":$typeWidth)>
  ];
}

def LoopOp : Plugin_Op<"loop", [NoSideEffect]> {
  let summary = "loop operation";
  let description = [{
        TODO.
  }];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<UI32Attr>:$index,
                       OptionalAttr<UI64Attr>:$innerLoopId,
                       OptionalAttr<UI64Attr>:$outerLoopId,
                       OptionalAttr<UI32Attr>:$numBlock);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "uint32_t":$index,
                      "uint64_t":$innerLoopId, "uint64_t":$outerLoopId,
                      "uint32_t":$numBlock)>
  ];
  let extraClassDeclaration = [{
    mlir::Block* GetHeader();
    mlir::Block* GetLatch();
    std::pair<mlir::Block*, mlir::Block*> GetSingleExit();
    void Delete();
    LoopOp GetInnerLoop();
    LoopOp GetOuterLoop();
    bool IsBlockInside(uint64_t);
    bool IsLoopFather(mlir::Block*);
    std::vector<std::pair<mlir::Block*, mlir::Block*> > GetExitEdges();
    std::vector<mlir::Block*> GetLoopBody(); 
    void AddLoop(uint64_t, uint64_t);
  }];
}

def CallOp : Plugin_Op<"call", [
    DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "call operation";
  let description = [{
    CallOp represent calls to a user defined function that needs to
    be specialized for the shape of its arguments.
    The callee name is attached as a symbol reference via an attribute.
    The arguments list must match the arguments expected by the callee.
  }];
  let arguments = (ins UI64Attr:$id,
                       FlatSymbolRefAttr:$callee,
                       Variadic<AnyType>:$inputs);
  let results = (outs Optional<AnyType>:$result);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$callee,
                  "ArrayRef<Value>":$arguments)>
  ];
  let extraClassDeclaration = [{
    bool SetLHS(Value lhs);
  }];
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def PhiOp : Plugin_Op<"phi", [NoSideEffect]> {
  let summary = "phi op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       UI32Attr:$capacity,
                       UI32Attr:$nArgs,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "uint32_t":$capacity,
                  "uint32_t":$nArgs, "ArrayRef<Value>":$operands,
                  "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    Value GetResult();
    Value GetArgDef(int i) { return getOperand(i); }
  }];
}

def AssignOp : Plugin_Op<"assign", [NoSideEffect]> {
  let summary = "assign op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       IExprCodeAttr:$exprCode,
                       Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IExprCode":$exprCode,
                  "ArrayRef<Value>":$operands, "Type":$resultType)>
  ];
  let extraClassDeclaration = [{
    Value GetLHS() { return getOperand(0); }
    Value GetRHS1() { return getOperand(1); }
    Value GetRHS2() { return getOperand(2); }
  }];
}

def PlaceholderOp : Plugin_Op<"placeholder", [NoSideEffect]> {
  let summary = "PlaceHolder.";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id,
                       OptionalAttr<IDefineCodeAttr>:$defCode);
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "IDefineCode":$defCode, "Type":$retType)>
  ];
}

def BaseOp : Plugin_Op<"statement_base", [NoSideEffect]> {
  let summary = "Base operation, just like placeholder for statement.";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, StrAttr:$opCode);
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$opCode)>
  ];
}

// Terminators
// Opaque builder used for terminator operations that contain successors.

class Plugin_TerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    Plugin_Op<mnemonic, !listconcat(traits, [Terminator])>;

def FallThroughOp : Plugin_TerminatorOp<"fallthrough", [NoSideEffect]> {
  let summary = "FallThroughOp";
  let description = [{TODO}];
  let successors = (successor AnySuccessor:$dest);
  // for bb address
  let arguments = (ins UI64Attr:$address, UI64Attr:$destaddr);
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$address, "Block*":$dest, "uint64_t":$destaddr)>
  ];
}

def CondOp : Plugin_TerminatorOp<"condition", [NoSideEffect]> {
  let summary = "condition op";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$id, UI64Attr:$address,
                       IComparisonAttr:$condCode,
                       AnyType:$LHS, AnyType:$RHS,
                       UI64Attr:$tbaddr,
                       UI64Attr:$fbaddr,
                       OptionalAttr<TypeAttr>:$trueLabel,
                       OptionalAttr<TypeAttr>:$falseLabel);
  let successors = (successor AnySuccessor:$tb, AnySuccessor:$fb);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "uint64_t":$address, "IComparisonCode":$condCode,
                  "Value":$lhs, "Value":$rhs, "Block*":$tb, "Block*":$fb,
                  "uint64_t":$tbaddr, "uint64_t":$fbaddr, "Value":$trueLabel,
                  "Value":$falseLabel)>,
    // Only for server.
    OpBuilderDAG<(ins "IComparisonCode":$condCode,
                  "Value":$lhs, "Value":$rhs)>
  ];
  let extraClassDeclaration = [{
    Value GetLHS() { return getOperand(0); }
    Value GetRHS() { return getOperand(1); }
  }];
}

// todo: currently RetOp do not have a correct assemblyFormat
def RetOp : Plugin_TerminatorOp<"ret", [NoSideEffect]> {
  let summary = "RetOp";
  let description = [{TODO}];
  let arguments = (ins UI64Attr:$address); // for bb address
  let results = (outs AnyType);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$address)>
  ];
}

#endif // PLUGIN_OPS_TD