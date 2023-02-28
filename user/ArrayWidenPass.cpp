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
    This file contains the implementation of the ArrayWidenPass class.
*/

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include "PluginAPI/PluginServerAPI.h"
#include "PluginServer/PluginLog.h"
#include "PluginAPI/ControlFlowAPI.h"
#include "user/ArrayWidenPass.h"

namespace PluginOpt {
using std::string;
using std::vector;
using std::cout;
using namespace mlir;
using namespace mlir::Plugin;
using namespace PluginAPI;
using namespace PinServer;
using namespace std;

mlir::MLIRContext *context;
mlir::OpBuilder* opBuilder = nullptr;
std::map<Block*, Value> defs_map;
std::map<uint64_t, std::string> opNameMap;

static void PassManagerSetupFunc(void)
{
    printf("PassManagerSetupFunc in\n");
}

enum EDGE_FLAG {
    EDGE_FALLTHRU,
    EDGE_TRUE_VALUE,
    EDGE_FALSE_VALUE
};

struct edgeDef {
    Block *src;
    Block *dest;
    unsigned destIdx;
    enum EDGE_FLAG flag;
};
typedef struct edgeDef edge;
typedef struct edgeDef *edgePtr;

static vector<Block *> getPredecessors(Block *bb)
{
    vector<Block *> preds;
    for (auto it = bb->pred_begin(); it != bb->pred_end(); ++it) {
        Block *pred = *it;
        preds.push_back(pred);
    }
    return preds;
}

static unsigned getNumPredecessor(Block *bb)
{
    vector<Block *> preds = getPredecessors(bb);
    return preds.size();
}

static unsigned getIndexPredecessor(Block *bb, Block *pred)
{
    unsigned i;
    vector<Block *> preds = getPredecessors(bb);
    for (i = 0; i < getNumPredecessor(bb); i++) {
        if (preds[i] == pred) {
            break;
        }
    }
    return i;
}

static enum EDGE_FLAG GetEdgeFlag(Block *src, Block *dest)
{
    Operation *op = src->getTerminator();
    enum EDGE_FLAG flag;
    if (isa<FallThroughOp>(op)) {
        flag = EDGE_FALLTHRU;
    }
    if (isa<CondOp>(op)) {
        if (op->getSuccessor(0) == dest) {
            flag = EDGE_TRUE_VALUE;
        } else {
            flag = EDGE_FALSE_VALUE;
        }
    }
    return flag;
}


static vector<edge> GetPredEdges(Block *bb)
{
    unsigned i = 0;
    vector<edge> edges;
    for (auto it = bb->pred_begin(); it != bb->pred_end(); ++it) {
        Block *pred = *it;
        edge e;
        e.src = pred;
        e.dest = bb;
        e.destIdx = i;
        e.flag = GetEdgeFlag(bb, pred);
        edges.push_back(e);
        i++;
    }
    return edges;
}

static edge GetEdge(Block *src, Block *dest)
{
    vector<edge> edges = GetPredEdges(dest);
    edge e;
    for (auto elm : edges) {
        if (elm.src == src) {
            e = elm;
            break;
        }
    }
    return e;
}

static unsigned GetEdgeDestIdx(Block *src, Block *dest)
{
    edge e = GetEdge(src, dest);
    return e.destIdx;
}

static bool IsEqualEdge(edge e1, edge e2)
{
    if (e1.src == e2.src && e1.dest == e2.dest && e1.destIdx == e2.destIdx && e1.flag == e2.flag) {
        return true;
    }
    return false;
}

static IDefineCode getValueDefCode(Value v)
{
    IDefineCode rescode;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        rescode = ssaop.defCode().getValue();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        rescode = memop.defCode().getValue();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        rescode = constop.defCode().getValue();
    } else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        rescode = holderop.defCode().getValue();
    }
    return rescode;
}

static uint64_t getValueId(Value v)
{
    uint64_t resid;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        resid = ssaop.id();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        resid = memop.id();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        resid = constop.id();
    } else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        resid = holderop.id();
    }
    return resid;
}

static PluginIR::PluginTypeBase getPluginTypeofValue(Value v)
{
    PluginIR::PluginTypeBase type;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        type = ssaop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        type = memop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        type = constop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    } else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        type = holderop.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
    }
    return type;
}

static bool isValueExist(Value v)
{
    uint64_t vid = getValueId(v);
    if (vid != 0) {
        return true;
    }
    return false;
}

static bool isEqualValue(Value v1, Value v2)
{
    uint64_t v1id = getValueId(v1);
    uint64_t v2id = getValueId(v2);
    if (v1id != 0 && v2id != 0 && v1id == v2id) {
        return true;
    }
    return false;
}

static bool isSingleRhsAssignOp(Operation *op)
{
    if (!isa<AssignOp>(op)) {
        return false;
    }
    if (op->getNumOperands() == 2) {
        return true;
    }
    return false;
}

static bool isBinaryRhsAssignOp(Operation *op)
{
    if (!isa<AssignOp>(op)) {
        return false;
    }
    if (op->getNumOperands() == 3) {
        return true;
    }
    return false;
}

static IDefineCode getSingleRhsAssignOpCode(Operation *op)
{
    auto assignOp = dyn_cast<AssignOp>(op);
    Value v = assignOp.GetRHS1();
    return getValueDefCode(v);
}

static IExprCode getBinaryRhsAssignOpCode(Operation *op)
{
    auto assignOp = dyn_cast<AssignOp>(op);
    return assignOp.exprCode();
}

static int64_t getRealValueIntCST(Value v)
{
    auto constOp = dyn_cast<ConstOp>(v.getDefiningOp());
    return constOp.initAttr().cast<mlir::IntegerAttr>().getInt();
}

static Operation *getSSADefStmtofValue(Value v)
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

struct originLoopInfo {
    Value base;     /* The initial index of the array in the old loop.  */
    Value *baseptr;
    Value limit;    /* The limit index of the array in the old loop.  */
    Value *limitptr;
    Value arr1;     /* Array 1 in the old loop.  */
    Value *arr1ptr;
    Value arr2;	    /* Array 2 in the old loop.  */
    Value *arr2ptr;
    edge entryEdge; /* The edge into the old loop.  */
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
    /* Whether the marker has an initial value assigned to the array index.  */
    uint64_t step;
    /* The growth step of the loop induction variable.  */
};

typedef struct originLoopInfo originLoopInfo;

static originLoopInfo originLoop;

/* Return true if the loop has precisely one backedge.  */
static bool isLoopSingleBackedge(LoopOp loop)
{
    Block *latch = loop.GetLatch();
    unsigned numSucc = latch->getNumSuccessors();
    if (numSucc != 1) {
        return false;
    }
    
    Block *header = loop.GetHeader();
    Block *latchSuccBB = latch->getSuccessor(numSucc-1);
    
    if (latchSuccBB != header) {
        return false;
    }
    
    return true;
}

/* Return true if the loop has precisely one preheader BB.  */
static bool isLoopSinglePreheaderBB(LoopOp loop)
{
    Block *header = loop.GetHeader();
    if (getNumPredecessor(header) != 2) {
        return false;
    }
    
    vector<Block *> preds = getPredecessors(header);
    Block *headerPred1 = preds[0];
    Block *headerPred2 = preds[1];

    Block *latch = loop.GetLatch();
    if ((headerPred1 == latch && !loop.IsLoopFather(headerPred2))
        || (headerPred2 == latch && !loop.IsLoopFather(headerPred1))) {
        return true;
    }
    return false;
}

/* Initialize the originLoop structure.  */
static void InitOriginLoopStructure()
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

static vector<edge> getLoopExitEdges(LoopOp loop)
{
    vector<std::pair<Block *, Block *>> bbPairInfo = loop.GetExitEdges();
    vector<edge> edges;
    for (auto elm : bbPairInfo) {
        edge e = GetEdge(elm.first, elm.second);
        edges.push_back(e);
    }
    return edges;
}

/* Make sure the exit condition stmt satisfies a specific form.  */

static bool checkCondOp(Operation *op)
{
    if (!op) {
        return false;
    }
    if (!isa<CondOp>(op)) {
        return false;
    }

    auto cond = dyn_cast<CondOp>(op);

    if (cond.condCode() != IComparisonCode::ne && cond.condCode() != IComparisonCode::eq) {
        return false;
    }
    
    Value lhs = cond.GetLHS();
    Value rhs = cond.GetRHS();
    if (getValueDefCode(lhs) != IDefineCode::SSA || getValueDefCode(rhs) != IDefineCode::SSA) {
        return false;
    }

    return true;
}

/* Record the exit information in the original loop including exit edge,
   exit bb block, exit condition stmt,
   eg: exit_eX origin_exit_bbX cond_stmtX.  */

static bool recordOriginLoopExitInfo(LoopOp loop)
{
    bool found = false;
    Operation *op;

    if (originLoop.exitE1Ptr != nullptr || originLoop.exitBB1 != nullptr
        || originLoop.exitE2Ptr != nullptr || originLoop.exitBB2 != nullptr
        || originLoop.condOp1 != nullptr || originLoop.condOp2 != nullptr) {
        return false;
    }

    vector<edge> exitEdges = getLoopExitEdges (loop);
    if (exitEdges.empty()) {
        return false;
    }

    if (exitEdges.size() != 2) {
        return false;
    }
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
      && originLoop.condOp1 != nullptr && originLoop.condOp2 != nullptr) {
        found = true;
    }

    return found;
}

/* Get the edge that first entered the loop.  */

static edge getLoopPreheaderEdge(LoopOp loop)
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

    edge e = GetEdge(src, header);

    return e;
}

/* Returns true if t is SSA_NAME and user variable exists.  */

static bool isSSANameVar(Value v)
{
    if (!isValueExist(v) || getValueDefCode(v) != IDefineCode::SSA) {
        return false;
    }
    auto ssaOp = dyn_cast<SSAOp>(v.getDefiningOp());
    uint64_t varid = ssaOp.nameVarId();
    if (varid != 0) {
        return true;
    }
    return false;
}

/* Returns true if t1 and t2 are SSA_NAME and belong to the same variable.  */

static bool isSameSSANameVar(Value v1, Value v2)
{
    if (!isSSANameVar (v1) || !isSSANameVar (v2)) {
        return false;
    }
    auto ssaOp1 = dyn_cast<SSAOp>(v1.getDefiningOp());
    auto ssaOp2 = dyn_cast<SSAOp>(v2.getDefiningOp());
    uint64_t varid1 = ssaOp1.nameVarId();
    uint64_t varid2 = ssaOp2.nameVarId();
    if (varid1 == varid2) {
        return true;
    }
    return false;
}

/* Get origin loop induction variable upper bound.  */
static bool getIvUpperBound(CondOp cond)
{
    if (originLoop.limitptr != nullptr) {
        return false;
    }
    
    Value lhs = cond.GetLHS();
    Value rhs = cond.GetRHS();
    if (!getPluginTypeofValue(lhs).isa<PluginIR::PluginIntegerType>()
        || !getPluginTypeofValue(rhs).isa<PluginIR::PluginIntegerType>()) {
        return false;
    }
  
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
static bool checkUpdateStmt(Operation *op)
{
    if (!op || !isa<AssignOp>(op)) {
        return false;
    }
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
static bool getIvBase(CondOp cond)
{
    if (originLoop.baseptr != nullptr || originLoop.updateStmt != nullptr) {
        return false;
    }
    
    Value lhs = cond.GetLHS();
    Block *header = cond.getOperation()->getBlock();
    auto &opList = header->getOperations();
    for (auto it = opList.begin(); it != opList.end(); ++it) {
        Operation *op = &(*it);
        if (!isa<PhiOp>(op)) {
            continue;
        }
        auto phi = dyn_cast<PhiOp>(op);
        Value result = phi.GetResult();
        if (!isSameSSANameVar (result, lhs)) {
            continue;
        }
        Value base = phi.GetArgDef(originLoop.entryEdge.destIdx);
        if (!isSameSSANameVar (base, lhs)) {
            return false;
        }

        originLoop.base = base;
        originLoop.baseptr = &originLoop.base;
        vector<edge> edges = GetPredEdges(header);
        for (auto e : edges) {
            if (!IsEqualEdge(e, originLoop.entryEdge)) {
                Value ivAfter = phi.GetArgDef(e.destIdx);
                if (!isa<SSAOp>(ivAfter.getDefiningOp())) {
                    return false;
                }
                Operation *op = getSSADefStmtofValue(ivAfter);
                if (!checkUpdateStmt (op)) {
                    return false;
                }
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
static bool recordOriginLoopHeader(LoopOp loop)
{
    Block *header = loop.GetHeader();
    if (originLoop.entryEdgePtr != nullptr || originLoop.baseptr != nullptr
        || originLoop.updateStmt != nullptr || originLoop.limitptr != nullptr) {
        return false;
    }
    originLoop.entryEdge = getLoopPreheaderEdge (loop);
    originLoop.entryEdgePtr = &originLoop.entryEdge;
    auto &opList = header->getOperations();
    for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
        Operation *op = &(*it);
        if (isa<PhiOp, SSAOp, PlaceholderOp, ConstOp>(op)) {
            continue;
        }

        if (auto cond = dyn_cast<CondOp>(op)) {
            if (!getIvUpperBound(cond)) {
                return false;
            }
            if (!getIvBase(cond)) {
                return false;
            }
        } else if (auto assign = dyn_cast<AssignOp>(op)) {
            if (op != originLoop.updateStmt || !originLoop.existPrologAssgin) {
                return false;
            }
        } else {
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
static bool recordOriginLoopLatch(LoopOp loop)
{
    Block *latch = loop.GetLatch();
    Operation *op = latch->getTerminator();

    if (originLoop.existPrologAssgin) {
        if (isa<FallThroughOp>(op)) {
            return true;
        }
    }

    // Todo: else分支处理别的场景，待添加

    return false;
}

/* Returns true when the DEF_STMT corresponding to arg0 of the mem_ref tree
   satisfies the POINTER_PLUS_EXPR type.  */
static bool checkBodyMemRef(Value memRef)
{
    if (getValueDefCode(memRef) != IDefineCode::MemRef) {
        return false;
    }
    
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
static bool checkBodyPointerPlus(Operation *op, Value &tmpIndex)
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
                } else if (!originLoop.arr2ptr) {
                    originLoop.arr2 = rhs1;
                    originLoop.arr2ptr = &originLoop.arr2;
                    if (!isEqualValue(tmpIndex, rhs2)) {
                        return false;
                    }
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
static bool recordOriginLoopBody(LoopOp loop)
{
    Block *body = originLoop.condOp2->getBlock();
    map<Operation*, bool> visited;
    for (auto &op : *body) {
        visited[&op] = false;
    }

    Value condLhs = dyn_cast<CondOp>(originLoop.condOp2).GetLHS();
    Value condRhs = dyn_cast<CondOp>(originLoop.condOp2).GetRHS();
    if (!(getPluginTypeofValue(condLhs).isa<PluginIR::PluginIntegerType>())
        || !(getPluginTypeofValue(condRhs).isa<PluginIR::PluginIntegerType>())) {
        return false;
    }

    vector<Value> worklist;
    worklist.push_back(condLhs);
    worklist.push_back(condRhs);
    Value tmpIndex;
    visited[originLoop.condOp2] = true;

    while (!worklist.empty()) {
        Value v = worklist.back();
        worklist.pop_back();
        Operation *op = getSSADefStmtofValue(v);
        
        if (!op || op->getBlock() != body || !isa<AssignOp>(op)) {
            continue;
        }
        visited[op] = true;
        if (isSingleRhsAssignOp(op) && getSingleRhsAssignOpCode(op) == IDefineCode::MemRef) {
            auto assignopmem = dyn_cast<AssignOp>(op);
            Value memRef = assignopmem.GetRHS1();
            if (!checkBodyMemRef(memRef)) {
                return false;
            }
            auto memOp = dyn_cast<MemOp>(memRef.getDefiningOp());
            Value arg0 = memOp.GetBase();
            worklist.push_back(arg0); // memref arg0
        } else if (isBinaryRhsAssignOp(op) && getBinaryRhsAssignOpCode(op) == IExprCode::PtrPlus) {
            auto assignop2 = dyn_cast<AssignOp>(op);
            Value rhs2 = assignop2.GetRHS2();
            if (!checkBodyPointerPlus(op, tmpIndex)) {
                return false;
            }
            worklist.push_back(rhs2);
        } else if (isSingleRhsAssignOp(op) && getBinaryRhsAssignOpCode(op) == IExprCode::Nop) {
            auto assignop = dyn_cast<AssignOp>(op);
            Value rhs = assignop.GetRHS1();
            if (!isSameSSANameVar(rhs, originLoop.base)) {
                return false;
            }
            worklist.push_back(rhs);
        } else {
            return false;
        }
    }
    bool allvisited = true;
    if (allvisited) {
        return true;
    }
    
    return false;
}

/* Returns true only if the exit bb of the original loop is unique and its phi
   node parameter comes from the same variable.  */
static bool checkExitBB(LoopOp loop)
{
    if (originLoop.exitBB1 != originLoop.exitBB2
      || loop.IsBlockInside (originLoop.exitBB1)) {
        return false;
    }

    for (auto &op : *originLoop.exitBB1) {
        if (!isa<PhiOp>(op)) {
            continue;
        }
        auto phi = dyn_cast<PhiOp>(op);
        Value result = phi.GetResult();
        if (!isSameSSANameVar(result, originLoop.base)) {
            continue;
        }
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
static bool checkOriginLoopInfo(LoopOp loop)
{
    if (!checkExitBB (loop)) {
        return false;
    }

    if (getValueDefCode(originLoop.base) != IDefineCode::SSA) {
        return false;
    }

    if (getValueDefCode(originLoop.limit) != IDefineCode::SSA) {
        return false;
    }

    auto limitssaop = dyn_cast<SSAOp>(originLoop.limit.getDefiningOp());
    if (!limitssaop.readOnly().getValue()) {
        return false;
    }

    if (!getPluginTypeofValue(originLoop.arr1).isa<PluginIR::PluginPointerType>()
        || !getPluginTypeofValue(originLoop.arr2).isa<PluginIR::PluginPointerType>()) {
        return false;
    }

    auto arr1type = getPluginTypeofValue(originLoop.arr1).dyn_cast<PluginIR::PluginPointerType>();
    auto arr2type = getPluginTypeofValue(originLoop.arr2).dyn_cast<PluginIR::PluginPointerType>();
    if (!arr1type.isReadOnlyElem() || !arr2type.isReadOnlyElem()) {
        return false;
    }
    
    if (!arr1type.getElementType().isa<PluginIR::PluginIntegerType>()
        || !arr2type.getElementType().isa<PluginIR::PluginIntegerType>()) {
        return false;
    }
    
    auto elemTy1 = arr1type.getElementType().dyn_cast<PluginIR::PluginIntegerType>();
    auto elemTy2 = arr2type.getElementType().dyn_cast<PluginIR::PluginIntegerType>();
    if (elemTy1.getWidth() != 8 || elemTy2.getWidth() != 8) {
        return false;
    }
    
    return true;
}

// /* Record the useful information of the original loop and judge whether the
//    information meets the specified conditions.  */
static bool checkRecordLoopForm(LoopOp loop)
{
    if (!recordOriginLoopExitInfo (loop)) {
        printf ("\nFailed to record loop exit information.\n");
        return false;
    }

    if (!recordOriginLoopHeader (loop)) {
        printf ("\nFailed to record loop header information.\n");
        return false;
    }

    if (!recordOriginLoopLatch (loop)) {
        printf ("\nFailed to record loop latch information.\n");
        return false;
    }

    if (!recordOriginLoopBody (loop)) {
        printf ("\nFailed to record loop body information.\n");
        return false;
    }

    if (!checkOriginLoopInfo (loop)) {
        printf ("\nFailed to check origin loop information.\n");
        return false;
    }

    return true;
}

static bool determineLoopForm(LoopOp loop)
{
    if (loop.innerLoopIdAttr().getInt() != 0 && loop.numBlockAttr().getInt() != 3) {
        printf ("\nWrong loop form, there is inner loop or redundant bb.\n");
        return false;
    }

    if (loop.GetSingleExit().first || !loop.GetLatch()) {
        printf ("\nWrong loop form, only one exit or loop_latch does not exist.\n");
        return false;
    }

    if (!isLoopSingleBackedge(loop)) {
        printf ("\nWrong loop form, loop back edges are not unique.\n");
        return false;
    }

    if (!isLoopSinglePreheaderBB(loop)) {
        printf ("\nWrong loop form, loop preheader bb are not unique.\n");
        return false;
    }

    InitOriginLoopStructure();
    if (!checkRecordLoopForm(loop)) {
        return false;
    }
    return true;
}

static void update_loop_dominator(uint64_t dir, FunctionOp* funcOp)
{
    ControlFlowAPI cfAPI;
    mlir::Region &region = funcOp->bodyRegion();
    PluginServerAPI pluginAPI;

    for (auto &bb : region.getBlocks()) {
        uint64_t bbAddr = pluginAPI.FindBasicBlock(&bb);
        if (bbAddr == 0) {
            continue;
        }
        uint64_t i_bbAddr = cfAPI.GetImmediateDominator(dir, bbAddr);
        if (!i_bbAddr || &bb == originLoop.exitBB1) {
            cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(&bb), \
                cfAPI.RecomputeDominator(1, pluginAPI.FindBasicBlock(&bb)));
            continue;
        }
    }
}

static void remove_originLoop(LoopOp *loop, FunctionOp* funcOp)
{
    vector<mlir::Block*> body;
    ControlFlowAPI controlapi;
    PluginServerAPI pluginAPI;
    body = loop->GetLoopBody();
    unsigned n = loop->numBlockAttr().getInt();
    for (unsigned i = 0; i < n; i++) {
        controlapi.DeleteBlock(body[i], funcOp->idAttr().getInt(), pluginAPI.FindBasicBlock(body[i]));
    }
    loop->Delete();
}

static void create_prolog_bb(Block *prolog_bb, Block *after_bb, Block *dominator_bb,
    LoopOp *outer, edge entryEdge, FunctionOp *funcOp, Block *ftBB)
{
    mlir::Value lhs1;

    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(prolog_bb), outer->idAttr().getInt());

    FallThroughOp fp = llvm::dyn_cast<FallThroughOp>(entryEdge.src->back());
    pluginAPI.RedirectFallthroughTarget(fp, pluginAPI.FindBasicBlock(entryEdge.src),
        pluginAPI.FindBasicBlock(prolog_bb));
    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(prolog_bb), pluginAPI.FindBasicBlock(dominator_bb));

    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    lhs1 = baseSsa.Copy();

    opBuilder->setInsertionPointToStart(prolog_bb);
    llvm::SmallVector<mlir::Value> ops;
    if (originLoop.existPrologAssgin) {
        ops.push_back(lhs1);
        ops.push_back(originLoop.base);
        mlir::Attribute attr = opBuilder->getI64IntegerAttr(originLoop.step);
        mlir::Value step = ConstOp::CreateConst(*opBuilder, attr, originLoop.base.getType());
        ops.push_back(step);
        AssignOp opa = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Plus, prolog_bb);
    } else {
        ops.push_back(lhs1);
        ops.push_back(originLoop.base);
        AssignOp op = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Nop, prolog_bb);
    }
    opBuilder->create<FallThroughOp>(opBuilder->getUnknownLoc(), pluginAPI.FindBasicBlock(prolog_bb), ftBB);
    baseSsa.SetCurrentDef(lhs1);
    defs_map.emplace(prolog_bb, lhs1);
}

static void create_loop_pred_bb(
    Block *loop_pred_bb, Block* after_bb, Block* dominator_bb, LoopOp *outer, FunctionOp *funcOp, Block *ftBB)
{
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(loop_pred_bb), outer->idAttr().getInt());
    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(loop_pred_bb), pluginAPI.FindBasicBlock(dominator_bb));
    opBuilder->setInsertionPointToStart(loop_pred_bb);
    opBuilder->create<FallThroughOp>(opBuilder->getUnknownLoc(), pluginAPI.FindBasicBlock(loop_pred_bb), ftBB);
    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    defs_map.emplace(loop_pred_bb, baseSsa.GetCurrentDef());
}

static void create_align_loop_header(Block *align_loop_header, Block* after_bb,
    Block* dominator_bb, LoopOp *outer, FunctionOp *funcOp, Block* tb, Block* fb)
{
    CondOp cond_stmt;
    PhiOp phi;
    Value res;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;
    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    Value entry_node = baseSsa.GetCurrentDef();
    
    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(align_loop_header), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(align_loop_header), pluginAPI.FindBasicBlock(dominator_bb));
    phi = PhiOp::CreatePhi(nullptr, align_loop_header);
    cfAPI.CreateNewDef(entry_node, phi.getOperation());
    res = phi.GetResult();

    opBuilder->setInsertionPointToStart(align_loop_header);
    llvm::SmallVector<mlir::Value> ops;

    PluginIR::PluginTypeBase baseType = PluginIR::PluginIntegerType::get(context, 64,
        PluginIR::PluginIntegerType::Unsigned);
    ops.push_back(SSAOp::MakeSSA(*opBuilder, baseType));
    ops.push_back(res);
    AssignOp op1 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Nop, (align_loop_header));
    Value lhs1 = op1.GetLHS();
    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, lhs1.getType()));
    ops.push_back(lhs1);
    ops.push_back(ConstOp::CreateConst(*opBuilder, opBuilder->getI64IntegerAttr(8), lhs1.getType()));
    AssignOp op2 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Plus, (align_loop_header));
    Value lhs2 = op2.GetLHS();
    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, baseType));
    ops.push_back(originLoop.limit);
    AssignOp op3 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Nop, (align_loop_header));
    Value lhs3 = op3.GetLHS();
    ops.clear();
    ops.push_back(lhs2);
    ops.push_back(lhs3);

    cond_stmt = opBuilder->create<CondOp>(opBuilder->getUnknownLoc(), IComparisonCode::le,
        lhs2, lhs3, tb, fb, (align_loop_header));

    baseSsa.SetCurrentDef(res);
    defs_map.emplace(align_loop_header, res);
}

static void rewrite_add_phi_arg(Block* bb)
{
    ControlFlowAPI cf;
    PluginServerAPI pluginAPI;
    vector<PhiOp> phis = cf.GetAllPhiOpInsideBlock(bb);
    for (auto phi : phis) {
        Value res = phi.GetResult();
        vector<Block *>  bv = getPredecessors (bb);
        int j = 0;
        for (int i = bv.size()-1; i>=0; i--) {
            if (phi.GetArgDef(j++)) {
                continue;
            }

            Value var = (defs_map[bv[i]]);
            if (!isSameSSANameVar(var, res)) {
                continue;
            }
            phi.AddArg(var, bv[i], bb);
        }
    }
}

static LoopOp *init_new_loop(LoopOp *outer_loop, Block* header, Block* latch, FunctionOp* funcOp)
{
    PluginServerAPI pluginAPI;
    LoopOp *new_loop;
    LoopOp new_loopt = funcOp->AllocateNewLoop();
    new_loop = &new_loopt;
    pluginAPI.SetHeader(new_loop->idAttr().getInt(), pluginAPI.FindBasicBlock(header));
    pluginAPI.SetLatch(new_loop->idAttr().getInt(), pluginAPI.FindBasicBlock(latch));
    new_loop->AddLoop(outer_loop->idAttr().getInt(), funcOp->idAttr().getInt());

    return new_loop;
}

static void create_align_loop_body_bb(Block *align_loop_body_bb, Block* after_bb, Block* dominator_bb,
    LoopOp *outer, FunctionOp *funcOp, Block* tb, Block* fb)
{
    CondOp cond_stmt;
    Value lhs1, lhs2;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(align_loop_body_bb), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(align_loop_body_bb),
        pluginAPI.FindBasicBlock(dominator_bb));

    opBuilder->setInsertionPointToStart(align_loop_body_bb);
    llvm::SmallVector<mlir::Value> ops;

    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    PluginIR::PluginTypeBase sizeType = PluginIR::PluginIntegerType::get(context, 64,
        PluginIR::PluginIntegerType::Unsigned);

    ops.push_back(SSAOp::MakeSSA(*opBuilder, sizeType));

    ops.push_back(baseSsa.GetCurrentDef());
    AssignOp op1 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::Nop, (align_loop_body_bb));
    Value var = op1.GetLHS();
    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, originLoop.arr2.getType()));
    ops.push_back(originLoop.arr2);
    ops.push_back(var);
    AssignOp op2 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::PtrPlus, (align_loop_body_bb));
    lhs1 = op2.GetLHS();
    ops.clear();
    PluginIR::PluginTypeBase baseType = PluginIR::PluginIntegerType::get(context, 64,
        PluginIR::PluginIntegerType::Unsigned);
    PluginIR::PluginTypeBase pointerTy = PluginIR::PluginPointerType::get(context, baseType, 0);
    ops.push_back(SSAOp::MakeSSA(*opBuilder, baseType));
    mlir::Attribute asdada = opBuilder->getI64IntegerAttr(0);
    ops.push_back(pluginAPI.BuildMemRef(baseType, lhs1, ConstOp::CreateConst(*opBuilder, asdada, pointerTy)));
    AssignOp op3 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::UNDEF, (align_loop_body_bb));

    lhs1 = op3.GetLHS();
    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, originLoop.arr1.getType()));
    ops.push_back(originLoop.arr1);
    ops.push_back(var);
    AssignOp op4 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::PtrPlus, (align_loop_body_bb));
    lhs2 = op4.GetLHS();
    ops.clear();

    ops.push_back(SSAOp::MakeSSA(*opBuilder, baseType));
    ops.push_back(pluginAPI.BuildMemRef(
        baseType, lhs2, ConstOp::CreateConst(*opBuilder, opBuilder->getI64IntegerAttr(0), pointerTy)));

    AssignOp op5 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::UNDEF, (align_loop_body_bb));
    lhs2 = op5.GetLHS();

    cond_stmt = opBuilder->create<CondOp>(opBuilder->getUnknownLoc(),
    llvm::dyn_cast<CondOp>(originLoop.condOp2).condCode(), lhs1, lhs2, tb, fb, (align_loop_body_bb));
}

static void create_align_loop_latch(Block *align_loop_latch, Block* after_bb, Block* dominator_bb,
    LoopOp *outer, FunctionOp *funcOp, Block *ftBB)
{
    Value res;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;

    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    Value entry_node = baseSsa.GetCurrentDef();

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(align_loop_latch), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(align_loop_latch),
        pluginAPI.FindBasicBlock(dominator_bb));

    res = baseSsa.Copy();

    opBuilder->setInsertionPointToStart(align_loop_latch);
    llvm::SmallVector<mlir::Value> ops;
    ops.push_back(res);
    ops.push_back(entry_node);
    ops.push_back(ConstOp::CreateConst(*opBuilder, opBuilder->getI64IntegerAttr(8), entry_node.getType()));
    opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Plus, (align_loop_latch));
    opBuilder->create<FallThroughOp>(opBuilder->getUnknownLoc(), pluginAPI.FindBasicBlock(align_loop_latch), ftBB);
    defs_map.emplace(align_loop_latch, res);
}

static void create_align_loop_exit_bb(Block *align_loop_exit_bb, Block* after_bb, Block* dominator_bb,
    LoopOp *outer, FunctionOp *funcOp, Block *ftBB)
{
    CondOp cond_stmt;
    Value lhs1, lhs2;
    Value cond_lhs, cond_rhs;
    CallOp build_ctzll;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;
    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    Value entry_node = baseSsa.GetCurrentDef();

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(align_loop_exit_bb), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(align_loop_exit_bb),
        pluginAPI.FindBasicBlock(dominator_bb));

    cond_stmt = llvm::dyn_cast<CondOp>(after_bb->back());
    cond_lhs = cond_stmt.GetLHS();
    cond_rhs = cond_stmt.GetRHS();

    opBuilder->setInsertionPointToStart(align_loop_exit_bb);
    llvm::SmallVector<mlir::Value> ops;
    ops.push_back((SSAOp::MakeSSA(*opBuilder, cond_lhs.getType())));
    ops.push_back(cond_lhs);
    ops.push_back(cond_rhs);
    AssignOp op1 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::BitXOR, (align_loop_exit_bb));
    lhs1 = op1.GetLHS();
    ops.clear();
    ops.push_back(lhs1);
    build_ctzll = opBuilder->create<CallOp>(opBuilder->getUnknownLoc(), ops, (align_loop_exit_bb));
    PluginIR::PluginTypeBase intType =
        PluginIR::PluginIntegerType::get(context, 32, PluginIR::PluginIntegerType::Signed);
    lhs1 = SSAOp::MakeSSA(*opBuilder, intType);
    build_ctzll.SetLHS(lhs1);

    SSAOp lhs1Ssa = dyn_cast<mlir::Plugin::SSAOp>(lhs1.getDefiningOp());
    lhs2 = lhs1Ssa.Copy();
    ops.clear();
    ops.push_back(lhs2);
    ops.push_back(lhs1);
    ops.push_back(ConstOp::CreateConst(*opBuilder, opBuilder->getI64IntegerAttr(3), lhs1.getType()));
    opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Rshift, (align_loop_exit_bb));

    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, entry_node.getType()));
    ops.push_back(lhs2);
    AssignOp op2 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Nop, (align_loop_exit_bb));
    lhs1 = op2.GetLHS();
    SSAOp entrySsa = dyn_cast<mlir::Plugin::SSAOp>(entry_node.getDefiningOp());
    lhs2 = entrySsa.Copy();
    ops.clear();
    ops.push_back(lhs2);
    ops.push_back(lhs1);
    ops.push_back(entry_node);
    opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Plus, (align_loop_exit_bb));

    opBuilder->create<FallThroughOp>(opBuilder->getUnknownLoc(), pluginAPI.FindBasicBlock(align_loop_exit_bb), ftBB);

    defs_map.emplace(align_loop_exit_bb, lhs2);
}

static void create_epilogue_loop_header(Block *epilogue_loop_header, Block* after_bb, Block* dominator_bb,
    LoopOp *outer, FunctionOp *funcOp, Block* tb, Block* fb)
{
    CondOp cond_stmt;
    Value res;
    PhiOp phi;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;

    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    Value entry_node = baseSsa.GetCurrentDef();

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(epilogue_loop_header), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(epilogue_loop_header),
        pluginAPI.FindBasicBlock(dominator_bb));

    phi = PhiOp::CreatePhi(nullptr, epilogue_loop_header);
    cfAPI.CreateNewDef(entry_node, phi.getOperation());
    res = phi.GetResult();

    opBuilder->setInsertionPointToStart(epilogue_loop_header);

    cond_stmt = opBuilder->create<CondOp>(opBuilder->getUnknownLoc(),
        llvm::dyn_cast<CondOp>(originLoop.condOp1).condCode(), res, originLoop.limit, tb, fb, (epilogue_loop_header));

    baseSsa.SetCurrentDef(res);
    defs_map.emplace(epilogue_loop_header, res);
}

static void create_epilogue_loop_body_bb(Block *epilogue_loop_body_bb, Block* after_bb, Block* dominator_bb,
    LoopOp *outer, FunctionOp *funcOp, Block* tb, Block* fb)
{
    AssignOp g;
    CondOp cond_stmt;
    Value lhs1, lhs2, lhs3;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;

    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    Value entry_node = baseSsa.GetCurrentDef();

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(epilogue_loop_body_bb), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(epilogue_loop_body_bb),
        pluginAPI.FindBasicBlock(dominator_bb));

    opBuilder->setInsertionPointToStart(epilogue_loop_body_bb);
    llvm::SmallVector<mlir::Value> ops;

    PluginIR::PluginTypeBase sizeType = PluginIR::PluginIntegerType::get(context, 64,
        PluginIR::PluginIntegerType::Unsigned);
    ops.push_back(SSAOp::MakeSSA(*opBuilder, sizeType));
    ops.push_back(entry_node);
    AssignOp op1 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::Nop, (epilogue_loop_body_bb));
    lhs1 = op1.GetLHS();
    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, originLoop.arr1.getType()));
    ops.push_back(originLoop.arr1);
    ops.push_back(lhs1);
    AssignOp op2 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::PtrPlus, (epilogue_loop_body_bb));
    lhs2 = op2.GetLHS();

    ops.clear();
    PluginIR::PluginTypeBase charType = PluginIR::PluginIntegerType::get(context, 8,
        PluginIR::PluginIntegerType::Unsigned);
    ops.push_back(SSAOp::MakeSSA(*opBuilder, charType));
    ops.push_back(pluginAPI.BuildMemRef(charType, lhs2,
        ConstOp::CreateConst(*opBuilder, opBuilder->getI64IntegerAttr(0), lhs2.getType())));
    g = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::UNDEF, (epilogue_loop_body_bb));

    lhs2 = g.GetLHS();

    ops.clear();
    ops.push_back(SSAOp::MakeSSA(*opBuilder, originLoop.arr2.getType()));
    ops.push_back(originLoop.arr2);
    ops.push_back(lhs1);
    AssignOp op3 = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops,
        IExprCode::PtrPlus, (epilogue_loop_body_bb));
    lhs3 = op3.GetLHS();

    ops.clear();

    ops.push_back(SSAOp::MakeSSA(*opBuilder, charType));
    ops.push_back(pluginAPI.BuildMemRef(charType, lhs3, ConstOp::CreateConst(*opBuilder,
        opBuilder->getI64IntegerAttr(0), lhs3.getType())));
    g = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::UNDEF, (epilogue_loop_body_bb));

    Value res = g.GetLHS();
    cond_stmt = opBuilder->create<CondOp>(opBuilder->getUnknownLoc(),
        llvm::dyn_cast<CondOp>(originLoop.condOp1).condCode(), lhs2, res, tb, fb, (epilogue_loop_body_bb));

    defs_map.emplace(epilogue_loop_body_bb, baseSsa.GetCurrentDef());
}

static void create_epilogue_loop_latch(Block *epilogue_loop_latch,
    Block *after_bb, Block *dominator_bb, LoopOp *outer, FunctionOp *funcOp, Block *ftBB)
{
    Value res;
    ControlFlowAPI cfAPI;
    PluginServerAPI pluginAPI;
    SSAOp baseSsa = dyn_cast<mlir::Plugin::SSAOp>(originLoop.base.getDefiningOp());
    Value entry_node = baseSsa.GetCurrentDef();

    pluginAPI.AddBlockToLoop(pluginAPI.FindBasicBlock(epilogue_loop_latch), outer->idAttr().getInt());

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(epilogue_loop_latch),
        pluginAPI.FindBasicBlock(dominator_bb));

    SSAOp entrySsa = dyn_cast<mlir::Plugin::SSAOp>(entry_node.getDefiningOp());
    res = entrySsa.Copy();

    opBuilder->setInsertionPointToStart(epilogue_loop_latch);
    llvm::SmallVector<mlir::Value> ops;
    ops.push_back(res);
    ops.push_back(entry_node);
    ops.push_back(ConstOp::CreateConst(*opBuilder,
        opBuilder->getI64IntegerAttr(originLoop.step), entry_node.getType()));
    opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(), ops, IExprCode::Plus, (epilogue_loop_latch));

    opBuilder->create<FallThroughOp>(opBuilder->getUnknownLoc(), pluginAPI.FindBasicBlock(epilogue_loop_latch), ftBB);

    defs_map.emplace(epilogue_loop_latch, res);
}

static void create_new_loops(edge entryEdge, FunctionOp *funcOp)
{
    Block* prolog_bb;
    Block* align_loop_header, *align_loop_latch, *align_loop_body_bb;
    Block* align_pred_bb, *align_loop_exit_bb;
    Block* epilogue_loop_header, *epilogue_loop_latch, *epilogue_loop_body_bb;
    Block* epilogue_loop_pred_bb;
    LoopOp *align_loop;
    LoopOp *epilogue_loop;
    PluginServerAPI pluginAPI;
    ControlFlowAPI cfAPI;
    LoopOp outer = pluginAPI.GetBlockLoopFather(pluginAPI.FindBasicBlock(entryEdge.src));

    uint64_t bbAddr = cfAPI.CreateBlock(entryEdge.src, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(entryEdge.src));
    prolog_bb = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(prolog_bb, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(prolog_bb));
    align_pred_bb = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(align_pred_bb, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(align_pred_bb));
    align_loop_header = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(align_loop_header, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(align_loop_header));
    align_loop_body_bb = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(align_loop_body_bb, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(align_loop_body_bb));
    align_loop_latch = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(align_loop_body_bb, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(align_loop_body_bb));
    align_loop_exit_bb = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(align_loop_header, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(align_loop_header));
    epilogue_loop_pred_bb = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(epilogue_loop_pred_bb, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(epilogue_loop_pred_bb));
    epilogue_loop_header = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(epilogue_loop_header, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(epilogue_loop_header));
    epilogue_loop_body_bb = pluginAPI.FindBlock(bbAddr);

    bbAddr = cfAPI.CreateBlock(epilogue_loop_body_bb, funcOp->idAttr().getInt(),
        pluginAPI.FindBasicBlock(epilogue_loop_body_bb));
    epilogue_loop_latch = pluginAPI.FindBlock(bbAddr);

    create_prolog_bb(prolog_bb, entryEdge.src, entryEdge.src, &outer, entryEdge, funcOp, align_pred_bb);
    
    create_loop_pred_bb(align_pred_bb, prolog_bb, prolog_bb, &outer, funcOp, align_loop_header);

    create_align_loop_header(align_loop_header, align_pred_bb, align_pred_bb, &outer,
        funcOp, align_loop_body_bb, epilogue_loop_pred_bb);

    create_align_loop_body_bb(align_loop_body_bb, align_loop_header, align_loop_header, &outer,
        funcOp, align_loop_exit_bb, align_loop_latch);

    create_align_loop_latch(align_loop_latch, align_loop_body_bb, align_loop_body_bb, &outer,
        funcOp, align_loop_header);

    rewrite_add_phi_arg(align_loop_header);

    align_loop = init_new_loop(&outer, align_loop_header, align_loop_latch, funcOp);

    create_align_loop_exit_bb(align_loop_exit_bb, align_loop_body_bb, align_loop_body_bb, &outer,
        funcOp, originLoop.exitBB1);

    create_loop_pred_bb(epilogue_loop_pred_bb, align_loop_header, align_loop_header, &outer,
        funcOp, epilogue_loop_header);

    create_epilogue_loop_header(epilogue_loop_header, epilogue_loop_pred_bb, epilogue_loop_pred_bb, &outer,
        funcOp, epilogue_loop_body_bb, originLoop.exitBB1);

    create_epilogue_loop_body_bb(epilogue_loop_body_bb, epilogue_loop_header, epilogue_loop_header, &outer,
        funcOp, originLoop.exitBB1, epilogue_loop_latch);

    create_epilogue_loop_latch(epilogue_loop_latch, epilogue_loop_body_bb, epilogue_loop_body_bb, &outer,
        funcOp, epilogue_loop_header);

    rewrite_add_phi_arg(epilogue_loop_header);

    epilogue_loop = init_new_loop(&outer, epilogue_loop_header, epilogue_loop_latch, funcOp);

    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(originLoop.exitBB1),
        pluginAPI.FindBasicBlock(entryEdge.src));
    cfAPI.SetImmediateDominator(1, pluginAPI.FindBasicBlock(originLoop.exitBB2),
        pluginAPI.FindBasicBlock(entryEdge.src));

    rewrite_add_phi_arg(originLoop.exitBB1);
    rewrite_add_phi_arg(originLoop.exitBB2);

    cfAPI.RemoveEdge(pluginAPI.FindBasicBlock(originLoop.exitE1.src),
        pluginAPI.FindBasicBlock(originLoop.exitE1.dest));
    cfAPI.RemoveEdge(pluginAPI.FindBasicBlock(originLoop.exitE2.src),
        pluginAPI.FindBasicBlock(originLoop.exitE2.dest));
}

static void convertToNewLoop(LoopOp* loop, FunctionOp* funcOp)
{
    ControlFlowAPI cfAPI;
    create_new_loops(originLoop.entryEdge, funcOp);
    remove_originLoop(loop, funcOp);
    update_loop_dominator(1, funcOp);
    cfAPI.UpdateSSA();
    return;
}

static void ProcessArrayWiden(uint64_t *fun)
{
    std::cout << "Running first pass, awiden\n";

    PluginServerAPI pluginAPI;
    
    FunctionOp funcOp = pluginAPI.GetFunctionOpById((uint64_t)fun);
    if (funcOp == nullptr) return;

    context = funcOp.getOperation()->getContext();
    mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
    opBuilder = &opBuilder_temp;
    string name = funcOp.funcNameAttr().getValue().str();
    printf("Now process func : %s \n", name.c_str());
    vector<LoopOp> allLoop = funcOp.GetAllLoops();
    for (auto &loop : allLoop) {
        if (determineLoopForm(loop)) {
            printf("The loop form is success matched, and the loop can be optimized.\n");
            convertToNewLoop(&loop, &funcOp);
        }
    }
}

int ArrayWidenPass::DoOptimize(uint64_t *fun)
{
    ProcessArrayWiden(fun);
    return 0;
}
}
