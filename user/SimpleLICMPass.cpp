#include "PluginAPI/ControlFlowAPI.h"
#include "PluginAPI/PluginServerAPI.h"
#include "user/SimpleLICMPass.h"
#include "PluginAPI/DataFlowAPI.h"
#include "mlir/Support/LLVM.h"

namespace PluginOpt {
using std::string;
using std::vector;
using std::cout;
using std::endl;
using namespace mlir;
using namespace PluginAPI;

PluginServerAPI pluginAPI;
ControlFlowAPI cfAPI;
DataFlowAPI dfAPI;
vector<AssignOp> move_stmt;

std::map<mlir::Operation*, bool> visited;
std::map<mlir::Operation*, bool> not_move;

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

static vector<Block *> getPredecessors(Block *bb)
{
    vector<Block *> preds;
    for (auto it = bb->pred_begin(); it != bb->pred_end(); ++it) {
        Block *pred = *it;
        preds.push_back(pred);
    }
    return preds;
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

static uint64_t getValueId(Value v)
{
    uint64_t resid = 0;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        resid = ssaop.getId();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        resid = memop.getId();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        resid = constop.getId();
    } else if (auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp())){
        resid = holderop.getId();
    } else if (auto componentop = dyn_cast<ComponentOp>(v.getDefiningOp())){
        resid = componentop.getId();
    }  else if (auto declop = llvm::dyn_cast<DeclBaseOp>(v.getDefiningOp())) {
        return declop.getId();
    }
    return resid;
}

static IDefineCode getValueDefCode(Value v)
{
    IDefineCode rescode;
    if (auto ssaop = dyn_cast<SSAOp>(v.getDefiningOp())) {
        rescode = ssaop.getDefCode().value();
    } else if (auto memop = dyn_cast<MemOp>(v.getDefiningOp())) {
        rescode = memop.getDefCode().value();
    } else if (auto constop = dyn_cast<ConstOp>(v.getDefiningOp())) {
        rescode = constop.getDefCode().value();
    } else {
        auto holderop = dyn_cast<PlaceholderOp>(v.getDefiningOp());
        rescode = holderop.getDefCode().value();
    }
    return rescode;
}

static bool isValueExist(Value v)
{
    uint64_t vid = getValueId(v);
    if (vid != 0) {
        return true;
    }
    return false;
}

static bool isSSANameVar(Value v)
{
    if (!isValueExist(v) || getValueDefCode(v) != IDefineCode::SSA) {
        return false;
    }
    auto ssaOp = dyn_cast<SSAOp>(v.getDefiningOp());
    uint64_t varid = ssaOp.getNameVarId();
    if (varid != 0) {
        return true;
    }
    return false;
}

static Operation *getSSADefStmtofValue(Value v)
{
    if (!isa<SSAOp>(v.getDefiningOp())) {
        return NULL;
    }
    auto ssaOp = dyn_cast<SSAOp>(v.getDefiningOp());
    // uint64_t id = ssaOp.getId();
    // pluginAPI.DebugValue(id);
    Operation *op = ssaOp.GetSSADefOperation();
    if (!op || !isa<AssignOp, PhiOp>(op)) {
        return NULL;
    }
    return op;
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

void compute_invariantness(Block* bb)
{
    LoopOp loop_father = pluginAPI.GetBlockLoopFather(bb);
    if (!loop_father.getOuterLoopId().value()){
        return ;
    }
    // pluginAPI.DebugBlock(bb);
    uint64_t bbAddr = pluginAPI.FindBasicBlock(bb);
    // 处理bb中的phi语句
    vector<PhiOp> phis = cfAPI.GetAllPhiOpInsideBlock(bb);
    for (auto phi : phis) {
        Value result = phi.GetResult();
        uint64_t varId = getValueId(result);
        // pluginAPI.DebugOperation(phi.getId());
        int n_args = phi.getNArgs();
        if (n_args <= 2 && !pluginAPI.IsVirtualOperand(varId)) {
            for (int i = 0 ; i < n_args; i++) {
                Value v = phi.GetArgDef(i);
                if (isSSANameVar(v)) {
                    Operation* def = getSSADefStmtofValue(v);
                    if (!def) break;
                    Block *def_bb = def->getBlock();
                    if (def_bb == bb && visited.find(def) == visited.end()) {
                        pluginAPI.DebugOperation(phi.getId());
                        not_move[def] = true;
                        break;
                    }
                }
            }
        }
    }
    vector<mlir::Operation*> ops = cfAPI.GetAllOpsInsideBlock(bb);
    map<uint64_t, bool> isinvariant;
    vector<mlir::Value> variants;
    int i = 0 ;
    bool change = false;
    int n = ops.size();
    do {
        change = false;
        for (auto op : ops) {
            bool may_move = true;
            if (not_move.find(op)!=not_move.end()) continue;
            if(!isa<AssignOp>(op)) continue;
            visited[op] = true;
            auto assign = dyn_cast<AssignOp>(op);
            int num = assign.getNumOperands();
            Value lhs = assign.GetLHS();
            Value rhs1 = assign.GetRHS1();
            
            Value vdef = dfAPI.GetGimpleVdef(assign.getId());
            uint64_t vdef_id = getValueId(vdef);
            if(vdef_id) {
                variants.push_back(lhs);
                not_move[op]=true;
                change = true;
                continue;
            }
            vector<mlir::Value> vals = dfAPI.GetSsaUseOperand(assign.getId());
            for (auto val : vals) {
                Operation* def = getSSADefStmtofValue(val);
                if(!def) continue;
                Block *def_bb = def->getBlock();
                if(def_bb == bb && visited.find(def) == visited.end()) {
                    not_move[def] = true;
                    not_move[op] = true;
                    change = true;
                    continue;
                } else if (not_move.find(def)!=not_move.end()) {
                    not_move[op] = true;
                    change = true;
                    continue;
                }
            }
            for (auto v : variants) {
                if (dfAPI.RefsMayAlias(lhs, v, 1) || dfAPI.RefsMayAlias(rhs1, v, 1)) {
                    may_move = false;
                    break;
                }
            }
            if(num == 3 && may_move) {
                Value rhs2 = assign.GetRHS2();
                for (auto v : variants) {
                    if (dfAPI.RefsMayAlias(rhs2, v, 1)) {
                        may_move = false;
                        break;
                    }
                }
            }
            if(!may_move) {
                not_move[op] = true;
                change = true;
            }
        }
    } while(change);
    
    cout<<"move statements: "<<endl;
    for (auto op : ops) {
        if (not_move.find(op) != not_move.end()) continue;
        if(!isa<AssignOp>(op)) continue;
        auto assign = dyn_cast<AssignOp>(op);
        pluginAPI.DebugOperation(assign.getId());
        move_stmt.push_back(assign);
    }
    cout<<" "<<endl;
    cout<<move_stmt.size()<<endl;
    return ;
}

LoopOp get_outermost_loop(AssignOp assign)
{
    Block *bb = assign->getBlock();
    LoopOp loop = pluginAPI.GetBlockLoopFather(bb);
    LoopOp maxloop;
    vector<mlir::Value> vals = dfAPI.GetSsaUseOperand(assign.getId());
    for (auto val: vals) 
    {
        uint64_t id = getValueId(val);
        pluginAPI.DebugValue(id);
        Operation* defOp = getSSADefStmtofValue(val);
        if (!defOp) continue;
        cout<<"none"<<endl;
        Block *def_bb = defOp->getBlock();
        LoopOp def_loop = pluginAPI.GetBlockLoopFather(def_bb);
        maxloop = pluginAPI.FindCommonLoop(&loop, &def_loop);
        if (maxloop == def_loop) {
            maxloop = loop;
        }
    }
    
    return maxloop;
}

void move_worker(AssignOp assign)
{
    LoopOp level = get_outermost_loop(assign);
    if (!level) return ;
    edge e = getLoopPreheaderEdge(level);

    // TODO.
}
static void ProcessSimpleLICM(uint64_t fun)
{
    cout << "Running first pass, Loop Invariant Code Motion\n";

    PluginServerAPI pluginAPI;
    DataFlowAPI dfAPI;

    dfAPI.CalDominanceInfo(1, fun);
    mlir::Plugin::FunctionOp funcOp = pluginAPI.GetFunctionOpById(fun);
    if (funcOp == nullptr) return;

    mlir::MLIRContext * context = funcOp.getOperation()->getContext();
    mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
    mlir::OpBuilder* opBuilder = &opBuilder_temp;
    string name = funcOp.getFuncNameAttr().getValue().str();
    fprintf(stderr, "Now process func : %s \n", name.c_str());
    vector<LoopOp> allLoop = funcOp.GetAllLoops();
    for (auto &loop : allLoop) {
        Block *header = loop.GetHeader();
        // pluginAPI.DebugBlock(header);
        compute_invariantness(header);
    }
    for (auto stmt : move_stmt) {
        // TODO.
        move_worker(stmt);
    }

}
int SimpleLICMPass::DoOptimize(uint64_t fun)
{
    ProcessSimpleLICM(fun);
    return 0;
}

}
