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

   Author: Mingchuan Wu and Yancheng Li
   Create: 2022-08-18
   Description:
    This file contains the implementation of the User Init.
*/

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include "PluginAPI/PluginServerAPI.h"
#include "PluginServer/PluginLog.h"

using std::string;
using std::vector;
using std::cout;
using namespace mlir;
using namespace mlir::Plugin;
using namespace PluginAPI;
using namespace PinServer;
using namespace std;

static void UserOptimizeFunc(void)
{
    PluginServerAPI pluginAPI;
    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();
    int count = 0;
    for (size_t i = 0; i < allFunction.size(); i++) {
        if (allFunction[i].declaredInlineAttr().getValue())
            count++;
    }
    printf("declaredInline have %d functions were declared.\n", count);
}

static uint64_t getBlockAddress(mlir::Block* b)
{
    if (mlir::Plugin::CondOp oops = dyn_cast<mlir::Plugin::CondOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::FallThroughOp oops = dyn_cast<mlir::Plugin::FallThroughOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::RetOp oops = dyn_cast<mlir::Plugin::RetOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else {
        assert(false);
    }
}

static void printBlock(mlir::Block* b)
{
    printf("[bb%ld]", getBlockAddress(b));
    printf(": has op %ld\n", b->getOperations().size());
    for (unsigned i = 0; i < b->getNumSuccessors(); i++) {
        printf("  --> ");
        printf("[bb%ld]", getBlockAddress(b->getSuccessor(i)));
        printf("\n");
    }
}

static void LocalVarSummery(void)
{
    PluginServerAPI pluginAPI;
    vector<mlir::Plugin::FunctionOp> allFunction = pluginAPI.GetAllFunc();
    map<string, string> args = PluginServer::GetInstance()->GetArgs();
    for (size_t i = 0; i < allFunction.size(); i++) {
        uint64_t funcID = allFunction[i].idAttr().getValue().getZExtValue();
        printf("In the %ldth function:\n", i);
        vector<mlir::Plugin::LocalDeclOp> decls = pluginAPI.GetDecls(funcID);
        int64_t typeFilter = -1u;
        if (args.find("type_code") != args.end()) {
            typeFilter = (int64_t)pluginAPI.GetTypeCodeFromString(args["type_code"]);
        }
        for (size_t j = 0; j < decls.size(); j++) {
            auto decl = decls[j];
            string name = decl.symNameAttr().getValue().str();
            int64_t declTypeID = decl.typeIDAttr().getValue().getZExtValue();
            if (declTypeID == typeFilter) {
                printf("\tFind %ldth target type %s\n", j, name.c_str());
            }
        }
    }
}

static void PassManagerSetupFunc(void)
{
    printf("PassManagerSetupFunc in\n");
}

enum EDGE_FLAG
{
    EDGE_FALLTHRU,
    EDGE_TRUE_VALUE,
    EDGE_FALSE_VALUE
};

struct edgeDef
{
    Block *src;
    Block *dest;
    unsigned destIdx;
    enum EDGE_FLAG flag;
};
typedef struct edgeDef edge;
typedef struct edgeDef *edgePtr;

static vector<Block *>
getPredecessors (Block *bb)
{
    vector<Block *> preds;
    for (auto it = bb->pred_begin(); it != bb->pred_end(); ++it)
    {
        Block *pred = *it;
        preds.push_back(pred);
    }
    return preds;
}

static unsigned
getNumPredecessor (Block *bb)
{
    vector<Block *> preds = getPredecessors(bb);
    return preds.size();
}

static unsigned
getIndexPredecessor (Block *bb, Block *pred)
{
    unsigned i;
    vector<Block *> preds = getPredecessors(bb);
    for (i = 0; i < getNumPredecessor(bb); i++)
    {
        if (preds[i] == pred)
            break;   
    }
    return i;
}

static enum EDGE_FLAG
getEdgeFlag (Block *src, Block *dest)
{
    Operation *op = src->getTerminator();
    enum EDGE_FLAG flag;
    if (isa<FallThroughOp>(op))
    {
        flag = EDGE_FALLTHRU;
    }
    if (isa<CondOp>(op))
    {
        if (op->getSuccessor(0) == dest)
            flag = EDGE_TRUE_VALUE;
        else
            flag = EDGE_FALSE_VALUE;
    }
    return flag;
}


static vector<edge>
getPredEdges (Block *bb)
{
    unsigned i = 0;
    vector<edge> edges;
    for (auto it = bb->pred_begin(); it != bb->pred_end(); ++it)
    {
        Block *pred = *it;
        edge e;
        e.src = pred;
        e.dest = bb;
        e.destIdx = i;
        e.flag = getEdgeFlag(bb, pred);
        edges.push_back(e);
        i++;
    }
    return edges;
}

static edge
getEdge (Block *src, Block *dest)
{
    vector<edge> edges = getPredEdges (dest);
    edge e;
    for (auto elm : edges)
    {
        if (elm.src == src) {
            e = elm;
            break;
        }
    }
    return e;
}

static unsigned
getEdgeDestIdx (Block *src, Block *dest)
{
    edge e = getEdge(src, dest);
    return e.destIdx;
}

static bool
isEqualEdge (edge e1, edge e2)
{
    if (e1.src == e2.src && e1.dest == e2.dest && e1.destIdx == e2.destIdx
        && e1.flag == e2.flag)
        return true;
    return false;
}

static IDefineCode
getValueDefCode (Value v)
{
    IDefineCode rescode;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        rescode = ssaop.defCode().getValue();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        rescode = memop.defCode().getValue();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        rescode = constop.defCode().getValue();
    }else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        rescode = holderop.defCode().getValue();
    }
    // assert(rescode == IDefineCode::UNDEF);
    return rescode;
}

static uint64_t
getValueId (Value v)
{
    uint64_t resid;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        resid = ssaop.id();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        resid = memop.id();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        resid = constop.id();
    }else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        resid = holderop.id();
    }
    return resid;
}

static PluginIR::PluginTypeBase
getPluginTypeofValue (Value v)
{
    PluginIR::PluginTypeBase type;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        type = ssaop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        type = memop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        type = constop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    }else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        type = holderop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    }
    return type;
}

static bool
isValueExist (Value v)
{
    uint64_t vid = getValueId(v);
    if (vid != 0) {
        return true;
    }
    return false;
}

static bool
isEqualValue (Value v1, Value v2)
{
    uint64_t v1id = getValueId(v1);
    uint64_t v2id = getValueId(v2);
    if (v1id != 0 && v2id != 0 && v1id == v2id)
        return true;
    return false;
}

static bool
isSingleRhsAssignOp (Operation *op)
{
    if (!isa<AssignOp>(op))
        return false;
    if (op->getNumOperands() == 2)
        return true;
    return false;
}

static bool
isBinaryRhsAssignOp (Operation *op)
{
    if (!isa<AssignOp>(op))
        return false;
    if (op->getNumOperands() == 3)
        return true;
    return false;
}

static IDefineCode
getSingleRhsAssignOpCode (Operation *op)
{
    auto assignOp = dyn_cast<AssignOp>(op);
    Value v = assignOp.GetRHS1();
    return getValueDefCode(v);
}

static IExprCode
getBinaryRhsAssignOpCode (Operation *op)
{
    auto assignOp = dyn_cast<AssignOp>(op);
    return assignOp.exprCode();
}

static int64_t
getRealValueIntCST (Value v)
{
    auto constOp = dyn_cast<ConstOp>(v.getDefiningOp());
    return constOp.initAttr().cast<mlir::IntegerAttr>().getInt();
}

static Operation *
getSSADefStmtofValue (Value v)
{
    if (!isa<SSAOp>(v.getDefiningOp())) {
        return NULL;
    }
    auto ssaOp = dyn_cast<SSAOp>(v.getDefiningOp());
    Operation *op = ssaOp.GetSSADefOperation();
    if (!op || !isa<AssignOp, PhiOp>(op)) {
        return NULL;
    }
    return op;
}

struct originLoopInfo
{
    Value base;		/* The initial index of the array in the old loop.  */
    Value *baseptr;
    Value limit;		/* The limit index of the array in the old loop.  */
    Value *limitptr;
    Value arr1;		/* Array 1 in the old loop.  */
    Value *arr1ptr;
    Value arr2;		/* Array 2 in the old loop.  */
    Value *arr2ptr;
    edge entryEdge;	/* The edge into the old loop.  */
    edgePtr entryEdgePtr;
    Block *exitBB1;
    Block *exitBB2;
    edge exitE1;
    edge exitE2;
    edgePtr exitE1Ptr;
    edgePtr exitE2Ptr;
    Operation *condOp1;
    Operation *condOp2;
    Operation *updateStmt;
    bool existPrologAssgin;
			/* Whether the marker has an initial value assigned
			   to the array index.  */
    uint64_t step;
  			/* The growth step of the loop induction variable.  */    
};

typedef struct originLoopInfo originLoopInfo;

static originLoopInfo originLoop;

/* Return true if the loop has precisely one backedge.  */

static bool
isLoopSingleBackedge (LoopOp loop)
{
    Block *latch = loop.GetLatch();
    unsigned numSucc = latch->getNumSuccessors();
    if (numSucc != 1)
        return false;
    
    Block *header = loop.GetHeader();
    Block *latchSuccBB = latch->getSuccessor(numSucc-1);
    
    if (latchSuccBB != header)
        return false;
    
    return true;
}

/* Return true if the loop has precisely one preheader BB.  */

static bool
isLoopSinglePreheaderBB (LoopOp loop)
{
    Block *header = loop.GetHeader();
    if (getNumPredecessor(header) != 2)
        return false;
    
    vector<Block *> preds = getPredecessors(header);
    Block *headerPred1 = preds[0];
    Block *headerPred2 = preds[1];

    Block *latch = loop.GetLatch();
    if ((headerPred1 == latch && !loop.IsLoopFather(headerPred2))
        || (headerPred2 == latch && !loop.IsLoopFather(headerPred1)))
        return true;
    return false;
}

/* Initialize the originLoop structure.  */
static void
initOriginLoopStructure ()
{
  originLoop.baseptr = nullptr;
  originLoop.limitptr = nullptr;
  originLoop.arr1ptr = nullptr;
  originLoop.arr2ptr = nullptr;
  originLoop.exitE1Ptr = nullptr;
  originLoop.exitE2Ptr = nullptr;
  originLoop.exitBB1 = nullptr;
  originLoop.exitBB2 =nullptr;
  originLoop.entryEdgePtr = nullptr;
  originLoop.condOp1 = nullptr;
  originLoop.condOp2 = nullptr;
  originLoop.updateStmt = nullptr;
  originLoop.existPrologAssgin = false;
  originLoop.step = 0;
}

static vector<edge>
getLoopExitEdges (LoopOp loop)
{
    vector<std::pair<Block *, Block *>> bbPairInfo = loop.GetExitEdges();
    vector<edge> edges;
    for (auto elm : bbPairInfo) {
        edge e;
        e = getEdge(elm.first, elm.second);
        edges.push_back(e);
    }
    return edges;
}

/* Make sure the exit condition stmt satisfies a specific form.  */

static bool
checkCondOp (Operation *op)
{
    if (!op)
        return false;
    if (!isa<CondOp>(op))
        return false;

    auto cond = dyn_cast<CondOp>(op);

    if (cond.condCode() != IComparisonCode::ne
        && cond.condCode() != IComparisonCode::eq)
        return false;
    
    Value lhs = cond.GetLHS();
    Value rhs = cond.GetRHS();

    if (getValueDefCode(lhs) != IDefineCode::SSA
        || getValueDefCode(rhs) != IDefineCode::SSA)
        return false;

  return true;
}

/* Record the exit information in the original loop including exit edge,
   exit bb block, exit condition stmt,
   eg: exit_eX origin_exit_bbX cond_stmtX.  */

static bool
recordOriginLoopExitInfo (LoopOp loop)
{
    bool found = false;
    Operation *op;

    if (originLoop.exitE1Ptr != nullptr || originLoop.exitBB1 != nullptr
        || originLoop.exitE2Ptr != nullptr || originLoop.exitBB2 != nullptr
        || originLoop.condOp1 != nullptr || originLoop.condOp2 != nullptr)
        return false;

    vector<edge> exitEdges = getLoopExitEdges (loop);
    if (exitEdges.empty())
        return false;

    if (exitEdges.size() != 2)
        return false;
    for (auto e : exitEdges) {
        if (e.src == loop.GetHeader()) {
            originLoop.exitE1 = e;
            originLoop.exitE1Ptr = &originLoop.exitE1;
            originLoop.exitBB1 = e.dest;
            op = &(e.src->back());
            if (checkCondOp (op)) {
                CondOp cond = dyn_cast<CondOp>(op);
                originLoop.condOp1 = cond.getOperation();
            }
        } else {
            originLoop.exitE2 = e;
            originLoop.exitE2Ptr = &originLoop.exitE2;
            originLoop.exitBB2 = e.dest;
            op = &(e.src->back());
            if (checkCondOp (op)) {
                CondOp cond = dyn_cast<CondOp>(op);
                originLoop.condOp2 = cond.getOperation();
            }
        }
    }
    
    

  if (originLoop.exitE1Ptr != nullptr && originLoop.exitBB1 != nullptr
      && originLoop.exitE2Ptr != nullptr && originLoop.exitBB2 != nullptr
      && originLoop.condOp1 != nullptr && originLoop.condOp2 != nullptr)
    found = true;

  return found;
}

/* Get the edge that first entered the loop.  */

static edge
getLoopPreheaderEdge (LoopOp loop)
{
    Block *header = loop.GetHeader();
    vector<Block *> preds = getPredecessors(header);

    Block *src;
    for (auto bb : preds) {
        if (bb != loop.GetLatch()) {
            src = bb;
            break;
        }
    }

    edge e = getEdge(src, header);

    return e;
}

/* Returns true if t is SSA_NAME and user variable exists.  */

static bool
isSSANameVar (Value v)
{
    if (!isValueExist(v) || getValueDefCode(v) != IDefineCode::SSA)
        return false;
    auto ssaOp = dyn_cast<SSAOp>(v.getDefiningOp());
    uint64_t varid = ssaOp.nameVarId();
    if (varid != 0)
        return true;
    return false;
}

/* Returns true if t1 and t2 are SSA_NAME and belong to the same variable.  */

static bool
isSameSSANameVar (Value v1, Value v2)
{
    if (!isSSANameVar (v1) || !isSSANameVar (v2))
        return false;
    auto ssaOp1 = dyn_cast<SSAOp>(v1.getDefiningOp());
    auto ssaOp2 = dyn_cast<SSAOp>(v2.getDefiningOp());
    uint64_t varid1 = ssaOp1.nameVarId();
    uint64_t varid2 = ssaOp2.nameVarId();
    if (varid1 == varid2)
        return true;
    return false;
}

/* Get origin loop induction variable upper bound.  */

static bool
getIvUpperBound (CondOp cond)
{
    if (originLoop.limitptr != nullptr)
        return false;
    
    Value lhs = cond.GetLHS();
    Value rhs = cond.GetRHS();

    if (!getPluginTypeofValue(lhs).isa<PluginIR::PluginIntegerType>() || !getPluginTypeofValue(rhs).isa<PluginIR::PluginIntegerType>())
        return false;
  
    originLoop.limit = rhs;
    originLoop.limitptr = &originLoop.limit;
    if (originLoop.limitptr != nullptr) {
        return true;
    }

    return false;
}

/* Returns true only when the expression on the rhs code of stmt is PLUS_EXPR,
   rhs1 is SSA_NAME with the same var as originLoop base, and rhs2 is
   INTEGER_CST.  */

static bool
checkUpdateStmt (Operation *op)
{
    if (!op || !isa<AssignOp>(op))
        return false;
    auto assignOp = dyn_cast<AssignOp>(op);
    if (assignOp.exprCode() == IExprCode::Plus) {
        
        Value rhs1 = assignOp.GetRHS1();
        Value rhs2 = assignOp.GetRHS2();
        if (getValueDefCode(rhs1) == IDefineCode::SSA
            && getValueDefCode(rhs2) == IDefineCode::IntCST
            && isSameSSANameVar (rhs1, originLoop.base)) {
            auto constOp = dyn_cast<ConstOp>(rhs2.getDefiningOp());
            originLoop.step = constOp.initAttr().cast<mlir::IntegerAttr>().getInt();
            if (originLoop.step == 1) {
                return true;
            }
        }
    }
    return false;
}

/* Get origin loop induction variable initial value.  */

static bool
getIvBase (CondOp cond)
{
    if (originLoop.baseptr != nullptr || originLoop.updateStmt != nullptr)
        return false;
    
    Value lhs = cond.GetLHS();

    Block *header = cond.getOperation()->getBlock();
    
    auto &opList = header->getOperations();
    for (auto it = opList.begin(); it != opList.end(); ++it) {
        Operation *op = &(*it);
        if (!isa<PhiOp>(op))
            continue;
        auto phi = dyn_cast<PhiOp>(op);
        Value result = phi.GetResult();
        if (!isSameSSANameVar (result, lhs))
            continue;
        Value base = phi.GetArgDef(originLoop.entryEdge.destIdx);
        if (!isSameSSANameVar (base, lhs))
            return false;

        originLoop.base = base;
        originLoop.baseptr = &originLoop.base;
        vector<edge> edges = getPredEdges(header);
        for (auto e : edges) {

            if (!isEqualEdge(e, originLoop.entryEdge)) {
                Value ivAfter = phi.GetArgDef(e.destIdx);
                if (!isa<SSAOp>(ivAfter.getDefiningOp()))
                    return false;
                Operation *op = getSSADefStmtofValue(ivAfter);
                if (!checkUpdateStmt (op))
                    return false;
                originLoop.updateStmt = op;
                if (op->getBlock() == header && isEqualValue(ivAfter, lhs)) {
                    originLoop.existPrologAssgin = true;
                }
            }
        }
    }
    if (originLoop.baseptr != nullptr && originLoop.updateStmt != nullptr) {
        return true;
    }
    
    return false;
}

/* Record the upper bound and initial value of the induction variable in the
   original loop; When prolog_assign is present, make sure loop header is in
   simple form; And the interpretation of prolog_assign is as follows:
   eg: while (++len != limit)
	......
   For such a loop, ++len will be processed before entering header_bb, and the
   assign is regarded as the prolog_assign of the loop.  */

static bool
recordOriginLoopHeader (LoopOp loop)
{
    Block *header = loop.GetHeader();

    if (originLoop.entryEdgePtr != nullptr || originLoop.baseptr != nullptr
        || originLoop.updateStmt != nullptr || originLoop.limitptr != nullptr)
        return false;
    originLoop.entryEdge = getLoopPreheaderEdge (loop);
    originLoop.entryEdgePtr = &originLoop.entryEdge;
    auto &opList = header->getOperations();
    for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
        Operation *op = &(*it);
        if (isa<PhiOp, SSAOp, PlaceholderOp, ConstOp>(op))
            continue;

        if (auto cond = dyn_cast<CondOp>(op)) {
            if (!getIvUpperBound (cond))
                return false;
            if (!getIvBase (cond))
                return false;
        } else if (auto assign = dyn_cast<AssignOp>(op)) {

            if (op != originLoop.updateStmt || !originLoop.existPrologAssgin) {
                return false;
            }
        }
        else {
            return false;
        }
    }

   if (originLoop.entryEdgePtr != nullptr && originLoop.baseptr != nullptr
      && originLoop.updateStmt != nullptr && originLoop.limitptr != nullptr) {
        return true;
    }

  return false;
}

/* When prolog_assign does not exist, make sure that updateStmt exists in the
   loop latch, and its form is a specific form, eg:
   len_2 = len_1 + 1.  */

static bool
recordOriginLoopLatch (LoopOp loop)
{
    Block *latch = loop.GetLatch();
    Operation *op = latch->getTerminator();

    if (originLoop.existPrologAssgin) {
        if (isa<FallThroughOp>(op))
            return true;
    }

    // Todo: else分支处理别的场景，待添加

    return false;
}

/* Returns true when the DEF_STMT corresponding to arg0 of the mem_ref tree
   satisfies the POINTER_PLUS_EXPR type.  */

static bool
checkBodyMemRef (Value memRef)
{
    if(getValueDefCode(memRef) != IDefineCode::MemRef)
        return false;
    
    auto memOp = dyn_cast<MemOp>(memRef.getDefiningOp());
    Value realarg0 = memOp.GetBase();
    Value realarg1 = memOp.GetOffset();
    PluginServerAPI pluginapi;
    Value arg0 = pluginapi.ConfirmValue(realarg0);
    Value arg1 = pluginapi.ConfirmValue(realarg1);

    if (getPluginTypeofValue(arg0).isa<PluginIR::PluginPointerType>()
        && getValueDefCode(arg1) == IDefineCode::IntCST
        && getRealValueIntCST(arg1) == 0) {
            Operation *op = getSSADefStmtofValue(arg0);
            if (op && isBinaryRhsAssignOp(op) && getBinaryRhsAssignOpCode(op) == IExprCode::PtrPlus) {
                return true;
            }
        }
    return false;
}

/* Returns true if the rh2 of the current stmt comes from the base in the
   original loop.  */

static bool
checkBodyPointerPlus (Operation *op, Value &tmpIndex)
{
    auto assignOp = dyn_cast<AssignOp>(op);
    Value rhs1 = assignOp.GetRHS1();
    Value rhs2 = assignOp.GetRHS2();

    if (getPluginTypeofValue(rhs1).isa<PluginIR::PluginPointerType>()) {
        Operation *g = getSSADefStmtofValue(rhs2);
        if (g && isSingleRhsAssignOp(g) && getBinaryRhsAssignOpCode(g) == IExprCode::Nop) {
            auto nopOp = dyn_cast<AssignOp>(g);
            Value nopRhs = nopOp.GetRHS1();
            
            if (isSameSSANameVar(nopRhs, originLoop.base)) {
                if (!originLoop.arr1ptr) {
                    originLoop.arr1 = rhs1;
                    originLoop.arr1ptr = &originLoop.arr1;
                    tmpIndex = rhs2;
                } else if(!originLoop.arr2ptr) {
                originLoop.arr2 = rhs1;
                originLoop.arr2ptr = &originLoop.arr2;
                if (!isEqualValue(tmpIndex, rhs2))
                    return false;
                } else {
                    return false;
                }
                return true;
            }
        }
    }
    return false;
}


/* Record the array comparison information in the original loop, while ensuring
   that there are only statements related to cont_stmt in the loop body.  */

static bool
recordOriginLoopBody (LoopOp loop)
{
    Block *body = originLoop.condOp2->getBlock();
    
    map<Operation*, bool> visited;
    for (auto &op : *body) {
        visited[&op] = false;
    }

    Value condLhs = dyn_cast<CondOp>(originLoop.condOp2).GetLHS();
    Value condRhs = dyn_cast<CondOp>(originLoop.condOp2).GetRHS();
    if (!(getPluginTypeofValue(condLhs).isa<PluginIR::PluginIntegerType>())
        || !(getPluginTypeofValue(condRhs).isa<PluginIR::PluginIntegerType>()))
        return false;

    vector<Value> worklist;
    worklist.push_back(condLhs);
    worklist.push_back(condRhs);
    Value tmpIndex;
    visited[originLoop.condOp2] = true;

    while (!worklist.empty())
    {
        Value v = worklist.back();
        worklist.pop_back();
        Operation *op = getSSADefStmtofValue(v);
        
        if (!op || op->getBlock() != body || !isa<AssignOp>(op))
            continue;
        visited[op] = true;
        if (isSingleRhsAssignOp(op) && getSingleRhsAssignOpCode(op) == IDefineCode::MemRef) {
            auto assignopmem = dyn_cast<AssignOp>(op);
            Value memRef = assignopmem.GetRHS1();
            if (!checkBodyMemRef(memRef))
                return false;
            auto memOp = dyn_cast<MemOp>(memRef.getDefiningOp());
            Value arg0 = memOp.GetBase();
            worklist.push_back(arg0);//memref arg0
        }else if (isBinaryRhsAssignOp(op) && getBinaryRhsAssignOpCode(op) == IExprCode::PtrPlus) {
            auto assignop2 = dyn_cast<AssignOp>(op);
            Value rhs2 = assignop2.GetRHS2();
            if (!checkBodyPointerPlus(op, tmpIndex))
                return false;
            worklist.push_back(rhs2);
        }else if (isSingleRhsAssignOp(op) && getBinaryRhsAssignOpCode(op) == IExprCode::Nop){
            auto assignop = dyn_cast<AssignOp>(op);
            Value rhs = assignop.GetRHS1();
            if (!isSameSSANameVar(rhs, originLoop.base))
                return false;
            worklist.push_back(rhs);
        }else {
            return false;
        }
    }
    bool allvisited = true;
    if (allvisited)
        return true;
    
    return false;
}

/* Returns true only if the exit bb of the original loop is unique and its phi
   node parameter comes from the same variable.  */

static bool
checkExitBB (LoopOp loop)
{
  if (originLoop.exitBB1 != originLoop.exitBB2
      || loop.IsBlockInside (originLoop.exitBB1))
    return false;

    for (auto &op : *originLoop.exitBB1) {
        if (!isa<PhiOp>(op))
            continue;
        auto phi = dyn_cast<PhiOp>(op);
        Value result = phi.GetResult();
        if (!isSameSSANameVar(result, originLoop.base))
            continue;
        if (phi.nArgs() == 2) {
            Value arg0 = phi.GetArgDef(0);
            Value arg1 = phi.GetArgDef(1);
            if (isEqualValue (arg0, arg1)) {
                return true;
            }
        }
    }
    return false;
}

/* Make sure that the recorded originLoop information meets the
   relative requirements.  */

static bool
checkOriginLoopInfo (LoopOp loop)
{
    if (!checkExitBB (loop))
        return false;

    if (getValueDefCode(originLoop.base) != IDefineCode::SSA)
        return false;

    if (getValueDefCode(originLoop.limit) != IDefineCode::SSA)
        return false;

    auto limitssaop = dyn_cast<SSAOp>(originLoop.limit.getDefiningOp());

    if (!limitssaop.readOnly().getValue())
        return false;

    if (!getPluginTypeofValue(originLoop.arr1).isa<PluginIR::PluginPointerType>()
        || !getPluginTypeofValue(originLoop.arr2).isa<PluginIR::PluginPointerType>())
        return false;

    auto arr1type = getPluginTypeofValue(originLoop.arr1).dyn_cast<PluginIR::PluginPointerType>();
    auto arr2type = getPluginTypeofValue(originLoop.arr2).dyn_cast<PluginIR::PluginPointerType>();

    if (!arr1type.isReadOnlyElem() || !arr2type.isReadOnlyElem())
        return false;
    
    if (!arr1type.getElementType().isa<PluginIR::PluginIntegerType>()
        || !arr2type.getElementType().isa<PluginIR::PluginIntegerType>())
        return false;
    
    auto elemTy1 = arr1type.getElementType().dyn_cast<PluginIR::PluginIntegerType>();
    auto elemTy2 = arr2type.getElementType().dyn_cast<PluginIR::PluginIntegerType>();

    if (elemTy1.getWidth() != 8 || elemTy2.getWidth() != 8)
        return false;
    
    return true;
}

// /* Record the useful information of the original loop and judge whether the
//    information meets the specified conditions.  */

static bool
checkRecordLoopForm (LoopOp loop)
{
    if (!recordOriginLoopExitInfo (loop))
    {
        printf ("\nFailed to record loop exit information.\n");
        return false;
    }

    if (!recordOriginLoopHeader (loop))
    {
        printf ("\nFailed to record loop header information.\n");
        return false;
    }

    if (!recordOriginLoopLatch (loop))
    {
        printf ("\nFailed to record loop latch information.\n");
        return false;
    }

    if (!recordOriginLoopBody (loop))
    {
        printf ("\nFailed to record loop body information.\n");
        return false;
    }

    if (!checkOriginLoopInfo (loop))
    {
        printf ("\nFailed to check origin loop information.\n");
        return false;
    }

  return true;
}

static bool
determineLoopForm(LoopOp loop)
{
    if (loop.innerLoopIdAttr().getInt() != 0 && loop.numBlockAttr().getInt() != 3)
    {
        printf ("\nWrong loop form, there is inner loop or redundant bb.\n");
        return false;
    }

    if (loop.GetSingleExit().first || !loop.GetLatch())
    {
        printf ("\nWrong loop form, only one exit or loop_latch does not exist.\n");
        return false;
    }

    if (!isLoopSingleBackedge(loop))
    {
        printf ("\nWrong loop form, loop back edges are not unique.\n");
        return false;
    }

    if (!isLoopSinglePreheaderBB(loop))
    {
        printf ("\nWrong loop form, loop preheader bb are not unique.\n");
        return false;
    }

    initOriginLoopStructure();
    if (!checkRecordLoopForm(loop))
        return false;
    return true;
}

static void
ProcessArrayWiden(void)
{
    std::cout << "Running first pass, awiden\n";

    PluginServerAPI pluginAPI;
    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();

    for (auto &funcOp : allFunction) {
        string name = funcOp.funcNameAttr().getValue().str();
        printf("Now process func : %s \n", name.c_str());
        vector<LoopOp> allLoop = funcOp.GetAllLoops();
        for (auto &loop : allLoop) {
            if (determineLoopForm(loop)) {
                printf("The %ldth loop form is success matched, and the loop can be optimized.\n", loop.indexAttr().getInt());
                return;
            }
        }
    }
}

void RegisterCallbacks(void)
{
    // PluginServer::GetInstance()->RegisterUserFunc(HANDLE_BEFORE_IPA, UserOptimizeFunc);
    // PluginServer::GetInstance()->RegisterUserFunc(HANDLE_BEFORE_IPA, LocalVarSummery);
    ManagerSetupData setupData;
    setupData.refPassName = PASS_PHIOPT;
    setupData.passNum = 1;
    setupData.passPosition = PASS_INSERT_AFTER;
    PluginServer::GetInstance()->RegisterPassManagerSetup(HANDLE_MANAGER_SETUP, setupData, ProcessArrayWiden);
}
