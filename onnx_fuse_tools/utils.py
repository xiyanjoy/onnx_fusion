import queue
import copy
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import torch
from tqdm import tqdm

# Map of dtype name -> torch dtype
string_to_numpy_dtype_dict = {
    "uint8"      : np.uint8,
    "int8"       : np.int8,
    "int16"      : np.int16,
    "int32"      : np.int32,
    "int64"      : np.int64,
    "float16"    : np.float16,
    "float32"    : np.float32,
    "float64"    : np.float64,
    "complex64"  : np.complex64,
    "complex128" : np.complex128
}

# Map of dtype name -> torch dtype
string_to_torch_dtype_dict = {
    "uint8"      : torch.uint8,
    "int8"       : torch.int8,
    "int16"      : torch.int16,
    "int32"      : torch.int32,
    "int64"      : torch.int64,
    "float16"    : torch.float16,
    "float32"    : torch.float32,
    "float64"    : torch.float64,
    "complex64"  : torch.complex64,
    "complex128" : torch.complex128
}

'''
gNode is not the node in onnx_graphsurgen defination.
In this util, a gNode can be a gs.ir.tensor.Tensor or a gs.ir.node.Node.
'''

def onnx_pmx(onnxmodel):
    onnxmodel.opset_import.append(onnxmodel.opset_import[0])
    print(f'Add domain "pmx" wiht opset version 1')
    onnxmodel.opset_import[-1].domain = 'pmx'
    onnxmodel.opset_import[-1].version = 1
    if onnxmodel.ir_version > 8:
        onnxmodel.ir_version = 8
    return onnxmodel

def isTensor(gNode):
    return isinstance(gNode, gs.ir.tensor.Tensor)

def isVariable(gNode):
    return isinstance(gNode, gs.ir.tensor.Variable)

def isConstant(gNode):
    return isinstance(gNode, gs.ir.tensor.Constant)

def isNode(gNode):
    return isinstance(gNode, gs.ir.node.Node)

def getConstInput(node):
    assert isNode(node), "Node should be a gs.ir.node.Node"
    res = list()
    for input in node.inputs:
        if isinstance(input, gs.ir.tensor.Constant):
            res.append(input)
    return res

def getVariableInput(node):
    assert isNode(node), "Node should be a gs.ir.node.Node"
    res = list()
    for input in node.inputs:
        if isinstance(input, gs.ir.tensor.Variable):
            res.append(input)
    return res

def gNodeIsInList(gNode, list_gNode):
    for ele in list_gNode:
        if type(gNode) == type(ele) and gNode == ele:
            return True
    return False

def getChildren(gNode):
    return gNode.outputs

def getParents(gNode):
    return gNode.inputs

def searchgNodeFromChildren(root, expr, max_depth=10000, include_root=True):
    '''
    example:
    if max_depth=2
        
        depth                       
        0               root
                         /\ 
                        /  \ 
        1          gNode0  gNode1
                    /| \   /   \ 
                   / |  \ /     \ 
        2      gNode2| gNode3  gNode4
                  |  |           /
                  | /           /
                gNode5         /
    --------------------------------------
                             / 
        3                 (gNode6)

    res = [
        (root, 0), 
        (gNode0, 1), (gNode1, 1), 
        (gNode2, 2), (gNode5, 2), (gNode3, 2), (gNode4, 2)
    ] if all expr(gNodes)==True

    if include_root==False, (root, 0) won't in the res
    Breadth-first search
    '''
    assert max_depth >= 0, "[E] max_depth < 0 is not acceptable"
    q = queue.Queue()
    vis = list()
    res = list()
    depth = 0

    if not root is None:
        q.put(root)
        vis.append(root.name)
    while not q.empty():
        if depth > max_depth:
            break
        size = q.qsize()
        for _ in range(size):
            gNode = q.get()
            if expr(gNode):
                if depth > 0 or (depth == 0 and include_root):
                    res.append((gNode, depth))
            children = gNode.outputs
            for child in children:
                if not child.name in vis:   # else gs.ir.tensor.Tensor == gs.ir.node.Node will goes wrong
                    q.put(child)
                    vis.append(child.name)
        depth += 1

    return res

def searchgNodeFromParents(root, expr, max_depth=10000, include_root=True):
    assert max_depth >= 0, "[E] max_depth < 0 is not acceptable"
    q = queue.Queue()
    vis = list()
    res = list()
    depth = 0

    if not root is None:
        q.put(root)
        vis.append(root.name)
    while not q.empty():
        if depth > max_depth:
            break
        size = q.qsize()
        for _ in range(size):
            gNode = q.get()
            if expr(gNode):
                if depth > 0 or (depth == 0 and include_root):
                    res.append((gNode, depth))
            children = gNode.inputs
            for child in children:
                if not child.name in vis:   # else gs.ir.tensor.Tensor == gs.ir.node.Node will goes wrong
                    q.put(child)
                    vis.append(child.name)
        depth += 1

    return res

def subGraphPatternMatching(
        graph,
        gNode_feature,
        edges,
        max_depth=10000
):
    assert len(gNode_feature) == len(edges), "[E] len(pattern) != len(edges)"
    path = [None for _ in range(len(gNode_feature))]
    res = list()
    adj_matrix_for_search_result = [[None for _ in range(len(gNode_feature))] for _ in range(len(gNode_feature))] 

    def dfs(gNode, cur_id, father_id):
        if path[cur_id] is None:
            if gNodeIsInList(gNode, path):
                # one gNode can't exist in path twice
                return
            
            # make sure it's connected to its father node
            for start in range(len(edges)):
                if father_id != -1 and father_id == start:
                    continue
                for end in edges[start]:
                    if end == cur_id and not path[start] is None:
                        if adj_matrix_for_search_result[start][cur_id] is None:
                            tmp = searchgNodeFromChildren(
                                path[start], 
                                gNode_feature[cur_id],
                                max_depth=max_depth,
                                include_root=False
                            )
                            adj_matrix_for_search_result[start][cur_id] = [ele[0] for ele in tmp]
                        if not gNodeIsInList(gNode, adj_matrix_for_search_result[start][cur_id]):
                            return
                        
            # make sure it's connected to its child node
            for next_id in edges[cur_id]:
                if not path[next_id] is None:
                    if adj_matrix_for_search_result[next_id][cur_id] is None:
                        tmp = searchgNodeFromParents(
                            path[next_id], 
                            gNode_feature[cur_id],
                            max_depth=max_depth,
                            include_root=False
                        )
                        adj_matrix_for_search_result[next_id][cur_id] = [ele[0] for ele in tmp]
                    if not gNodeIsInList(gNode, adj_matrix_for_search_result[next_id][cur_id]):
                        return

            path[cur_id] = gNode
            first_none = -1
            for next_id in edges[cur_id]:
                if path[next_id] is None:
                    first_none = next_id
                    break
            if first_none != -1:
                possible_next_gNode = searchgNodeFromChildren(
                    gNode, 
                    gNode_feature[first_none],
                    max_depth=max_depth,
                    include_root=False
                )
                for next_gNode in possible_next_gNode:
                    if gNode_feature[first_none](next_gNode[0]):
                        dfs(next_gNode[0], first_none, cur_id)
            else:
                for next_id in range(len(gNode_feature)):
                    if path[next_id] is None:
                        startFromRoot(next_id)
                        break
                    if next_id == (len(gNode_feature) - 1):
                        res.append(path.copy())
            path[cur_id] = None
            adj_matrix_for_search_result[cur_id] = [None for _ in range(len(gNode_feature))]
        else:
            if (gNode == path[cur_id]):
                for next_id in range(len(gNode_feature)):
                    if path[next_id] is None:
                        startFromRoot(next_id)
                        break
                    if next_id == (len(gNode_feature) - 1):
                        res.append(path.copy())

    def startFromRoot(cur_id):
        for gNode in graph.nodes + list(graph.tensors().values()):
            if gNode_feature[cur_id](gNode):
                dfs(gNode, cur_id, -1)

        return

    startFromRoot(0)
    return res

def traversalPatternMatching(
        graph, 
        s_expr, 
        r_expr,
        e_expr,
        direction="down",
        max_depth=10000
    ):
    res = list()
    path = list()

    def dfs(cur, depth):
        if depth > max_depth:
            return
        children = list()
        if direction=="down":
            children = cur.outputs
        elif direction=="up":
            children = cur.inputs

        if len(children) == 0:
            return

        for child in children:
            if e_expr(child, path):
                path.append(child)
                res.append(path.copy())
                path.pop()
                continue
            if r_expr(child, path):
                path.append(child)
                dfs(child, depth + 1)
                path.pop()

    for s_gNode in graph.nodes + list(graph.tensors().values()):
        if not s_gNode is None and s_expr(s_gNode):
            path.append(s_gNode)
            dfs(s_gNode, 0)
            path.pop()

    return res

def setShapeOrIndex(graph):
    def setAttrs(path):
        for gNode in path:
            if isNode(gNode):
                gNode.shape_or_index = True
            elif isTensor(gNode):
                gNode.shape_or_index = True

    def end_expr(gNode, path):
        if not isNode(gNode):
            return False
        assert isNode(gNode), "gNode isn't node"
        judge0 =  gNode.op in ["ConstantOfShape", "Range"]
        judge1 = gNode.op in {"Reshape", "Expand"} and gNode.inputs[1] == path[-1]
        judge2 = gNode.op == "Slice" and gNode.inputs[0] != path[-1]
        judge3 = \
            gNode.op == ["Add", "Sub", "Mul", "Div"] and \
            len(getConstInput(gNode)) == 0 and \
            gNode.output[0].shape != path[-1].shape
        return judge0 or judge1 or judge2 and judge3

    possible_path = traversalPatternMatching(
        graph,
        lambda x: isNode(x) and x.op == "Shape",
        lambda x, path: True,
        end_expr
    )
    for path in possible_path:
        setAttrs(path[: -1])

    for gNode in graph.nodes + list(graph.tensors().values()):
        if hasattr(gNode, "shape_or_index") and gNode.shape_or_index:
            gNode.name += "(shape_or_index)"
        else:
            gNode.shape_or_index = False

def merge_SelfAttention(graph, max_depth=30):
    op_name = "SelfAttention"
    print(f"[I] Merging {op_name}")
    setShapeOrIndex(graph)
    gNode_feature = [
        lambda x: isNode(x) and x.op == "Softmax" and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 0,
        lambda x: isVariable(x) and len(x.outputs) >= 3 and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 0,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
        lambda x: isVariable(x) and hasattr(x, "shape_or_index") and not x.shape_or_index,
    ]
    edges = [
        [6],
        [6],
        [4],
        [4],
        [0],
        [1, 2, 3],
        [7],
        [8],
        []
    ]
    output_edge = 7
    patterns = subGraphPatternMatching(
        graph,
        gNode_feature = gNode_feature,
        edges = edges,
        max_depth=max_depth
    )

    def args_process(args_relay, new_name):
        res = {
            "attn_type": 0,
            "l2norm": 0,
            "num_heads": None,
            "input": None,
            "key_padding_mask": None,
            "in_project_weight": None,
            "in_project_bias": None,
            "out_project_weight": None,
            "out_project_bias": None,
            "gamma": None,
            "temperature": None,
            "output": None
        }
        in_q_project_weight = args_relay["in_q_project_weight"].values
        in_k_project_weight = args_relay["in_k_project_weight"].values
        in_v_project_weight = args_relay["in_v_project_weight"].values
        in_project_weight = gs.Constant(
            f"{new_name}/in_project_weight", 
            values=np.concatenate([
                in_q_project_weight, 
                in_k_project_weight, 
                in_v_project_weight
            ], axis=-1)
        )

        in_project_bias = None
        if "in_q_project_bias" in args_relay.keys():
            in_q_project_bias = args_relay["in_q_project_bias"].values
            in_k_project_bias = args_relay["in_k_project_bias"].values
            in_v_project_bias = args_relay["in_v_project_bias"].values
            in_project_bias = gs.Constant(
                f"{new_name}/in_project_bias", 
                values=np.concatenate([
                    in_q_project_bias, 
                    in_k_project_bias, 
                    in_v_project_bias
                ], axis=-1))
        else:
            in_project_bias = gs.Constant(
                f"{new_name}/in_project_bias", 
                values=np.array([], dtype=in_q_project_weight.dtype)
            )

        for key in res.keys():
            if key == "in_project_weight":
                res[key] = in_project_weight
                continue
            if key == "in_project_bias":
                res[key] = in_project_bias
                continue
            if key in args_relay.keys():
                res[key] = args_relay[key]
            elif res[key] is None:
                res[key] = gs.Constant(
                    f"{new_name}/{key}", 
                    values=np.array([], dtype=in_q_project_weight.dtype)
                )

        return res

    # 5->1, 2, 3
    def judge_5_123(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode):
                if gNode.op == "Cast":
                    return False
                else:
                    compute_nodes.append(gNode)
        return len(compute_nodes) == 1

    def pass_5_123(path):
        return {"input": path[0]}
    
    # 1->6
    def judge_1_6(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "MatMul"],
        ]
        return judge0

    def pass_1_6(path):
        res = {"in_v_project_weight": getConstInput(path[0])[0]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["in_v_project_bias"] = getConstInput(gNode)[0]
                break
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Reshape":
                res["num_heads"] = path[i + 1].shape[-2]
                break
        return res
    
    # 2->4
    def judge_2_4(paths):
        if len(paths) != 1:
            return False
        compute_Nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_Nodes.append(gNode)
        judge0 = [node.op for node in compute_Nodes] in [
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Mul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Mul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Mul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Add", "Mul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "Mul", "MatMul"],
        ]

        judge1 = None
        shape0 = None
        shape1 = None
        if judge0:
            for node in compute_Nodes:
                if node.op == "Reshape":
                    shape0 = node.outputs[0].shape
            shape1 = compute_Nodes[-2].outputs[0].shape
            judge1 = (shape0[-1] == shape1[-1])
        
        return judge0 and judge1
    
    def pass_2_4(path):
        res = {"in_q_project_weight": getConstInput(path[0])[0]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["in_q_project_bias"] = getConstInput(gNode)[0]
                break
        return res

    # 3->4
    def judge_3_4(paths):
        if len(paths) != 1:
            return False
        compute_Nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_Nodes.append(gNode)
        judge0 = [node.op for node in compute_Nodes] in [
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "Transpose", "MatMul"],
        ]

        judge1 = None
        shape0 = None
        shape1 = None
        if judge0:
            for node in compute_Nodes:
                if node.op == "Reshape":
                    shape0 = node.outputs[0].shape
            shape1 = compute_Nodes[-2].outputs[0].shape
            judge1 = (shape0[-1] == shape1[-2])
        
        return judge0 and judge1
    
    def pass_3_4(path):
        res = {"in_k_project_weight": getConstInput(path[0])[0]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["in_k_project_bias"] = getConstInput(gNode)[0]
                break
        return res
    
    # 4->0
    def judge_4_0(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for i, gNode in enumerate(paths[0]):
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["MatMul", "Reshape", "Add", "Reshape", "Softmax"],
            ["MatMul", "Div", "Add", "Softmax"],
            ["MatMul", "Mul", "Add", "Softmax"],
            ["MatMul", "Softmax"]
        ]
        return judge0
    
    def pass_4_0(path):
        res = {}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                index = 0 if gNode.inputs[1] == path[i - 1] else 1
                res["key_padding_mask"] = gNode.inputs[index]
                break
        return res

    # 0->6
    def judge_0_6(paths):
        if len(paths) != 1:
            return False
        compute_Nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast" and gNode.op != "Dropout":
                compute_Nodes.append(gNode)
        judge0 = [node.op for node in compute_Nodes] in [
            ["Softmax", "MatMul"],
        ]
        return judge0
    
    # 6->7
    def judge_6_7(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["MatMul", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"]
        ]
        return judge0

    # 7->8
    def judge_7_8(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = True
        if [node.op for node in compute_nodes] == ["MatMul", "Add"]:
            judge0 = getConstInput(compute_nodes[0])[0].shape[-1] == \
                getConstInput(compute_nodes[1])[0].shape[-1]
        elif [node.op for node in compute_nodes] == ["MatMul"]:
            if len(paths[0][-1].outputs) == 1 and paths[0][-1].outputs[0].op == "Add":
                add = paths[0][-1].outputs[0]
                if getConstInput(compute_nodes[0])[0].shape[-1] == \
                    getConstInput(add)[0].shape[-1]:
                    judge0 = False
        else:
            judge0 = False
        judge1 = not (isNode(paths[0][-2]) and paths[0][-2].op =="Cast")
        return judge0 and judge1
    
    def pass_7_8(path):
        res = {"out_project_weight": path[0].inputs[1]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["out_project_bias"] = getConstInput(gNode)[0]
                break
        res["output"] = path[-1]
        return res

    judges = [
        [judge_0_6],
        [judge_1_6],
        [judge_2_4],
        [judge_3_4],
        [judge_4_0],
        [judge_5_123, judge_5_123, judge_5_123],
        [judge_6_7],
        [judge_7_8],
        []
    ]
    passes = [
        [None],
        [pass_1_6],
        [pass_2_4],
        [pass_3_4],
        [pass_4_0],
        [pass_5_123, None, None],
        [None],
        [pass_7_8],
        []
    ]
    pattern_detail_list = list()
    print("patterns: {}".format(len(patterns)))
    for pattern in tqdm(patterns):
        label = True
        pattern_detail = copy.deepcopy(edges)
        for start in range(len(edges)):
            if not label:
                break
            for i, end in enumerate(edges[start]):
                if not label:
                    break
                paths = traversalPatternMatching(
                    graph,
                    lambda x: x.name == pattern[start].name,
                    lambda x, path: not x.shape_or_index,
                    lambda x, path: x.name == pattern[end].name,
                    max_depth=max_depth
                )
                label = label and judges[start][i](paths)
                pattern_detail[start][i] = paths[0]
        if label:
            pattern_detail_list.append(pattern_detail)
    
    # res = list()
    count = 0
    for pattern_detail in pattern_detail_list:
        args_relay = {} 
        for start in range(len(edges)):
            for i, end in enumerate(edges[start]):
                if not passes[start][i] is None:
                    args_relay.update(passes[start][i](pattern_detail[start][i]))
        new_name = f"{op_name}_{count}"
        args_final = args_process(args_relay, new_name)

        SelfAttention_node = gs.Node(
            op=op_name,
            name=new_name,
            attrs={
                "num_heads": args_final["num_heads"], 
                "l2norm": args_final["l2norm"], 
                "attn_type": args_final["attn_type"]
            },
            inputs=[
                args_final["input"], 
                args_final["key_padding_mask"], 
                args_final["in_project_weight"], 
                args_final["in_project_bias"], 
                args_final["out_project_weight"], 
                args_final["out_project_bias"], 
                args_final["gamma"], 
                args_final["temperature"]
            ],
            outputs=[args_final["output"]],
        )
        SelfAttention_node.domain = "pmx"
        graph.nodes.append(SelfAttention_node)
        count += 1
        input_name = args_final["input"].name
        output_name = args_final["output"].name
        print(f"[I]    find {new_name} between {input_name} -> {output_name}")
        # res.append((pattern_detail, args_final))
        pattern_detail[output_edge][0][-2].outputs = []
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_ContextAttention(graph, max_depth=30):
    op_name = "ContextAttention"
    print(f"[I] Merging {op_name}")
    setShapeOrIndex(graph)
            
    gNode_feature = [
        lambda x: isNode(x) and x.op == "Softmax" and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 0,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
        lambda x: isVariable(x) and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) >= 1,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) >= 1,
        lambda x: isNode(x) and x.op == "MatMul" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 0,
        lambda x: isTensor(x) and len(x.outputs) >= 2 and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isVariable(x) and len(x.outputs) >= 1 and hasattr(x, "shape_or_index") and not x.shape_or_index,
    ]
    edges = [
        [1],
        [2],
        [3],
        [],
        [1],
        [7],
        [7],
        [0],
        [4, 6],
        [5]
    ]
    output_edge = 2
    patterns = subGraphPatternMatching(
        graph,
        gNode_feature = gNode_feature,
        edges = edges,
        max_depth=max_depth
    )

    def args_process(args_relay, new_name):
        res = {
            "attn_type": 0,
            "l2norm": 0,
            "num_heads": None,
            "input": None,
            "context": None,
            "key_padding_mask": None,
            "in_project_weight_q": None,
            "in_project_weight_k": None,
            "in_project_weight_v": None,
            "out_project_weight": None,
            "out_project_bias": None,
            "gamma": None,
            "temperature": None,
            "output": None
        }
        dtype = args_relay["in_project_weight_q"].values.dtype

        for key in res.keys():
            if key in args_relay.keys():
                res[key] = args_relay[key]
            elif res[key] is None:
                res[key] = gs.Constant(
                    f"{new_name}/{key}", 
                    values=np.array([], dtype=dtype)
                )

        return res

    # 0->2
    def judge_8_46(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        return len(compute_nodes) == 1
    
    def pass_8_4(path):
        return {"context": path[0]}
    
    # 1->3, 4
    def judge_9_5(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        return len(compute_nodes) == 1

    def pass_9_5(path):
        return {"input": path[0]}

    # 2->7
    def judge_4_1(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "MatMul"],
        ]
        return judge0

    def pass_4_1(path):
        res = {"in_project_weight_v": getConstInput(path[0])[-1]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["in_project_bias_v"] = getConstInput(gNode)[0]
                break
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Reshape":
                res["num_heads"] = path[i + 1].shape[-2]
                break
        return res
    
    # 3->5
    def judge_5_7(paths):
        if len(paths) != 1:
            return False
        compute_Nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_Nodes.append(gNode)
        judge0 = [node.op for node in compute_Nodes] in [
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Mul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Mul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Mul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Add", "Mul", "Reshape", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "Mul", "MatMul"],
        ]

        judge1 = None
        shape0 = None
        shape1 = None
        if judge0:
            for node in compute_Nodes:
                if node.op == "Reshape":
                    shape0 = node.outputs[0].shape
            shape1 = compute_Nodes[-2].outputs[0].shape
            judge1 = (shape0[-1] == shape1[-1])
        
        return judge0 and judge1
    
    def pass_5_7(path):
        res = {"in_project_weight_q": getConstInput(path[0])[0]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["in_project_bias_q"] = getConstInput(gNode)[0]
                break
        return res

    # 4->5
    def judge_6_7(paths):
        if len(paths) != 1:
            return False
        compute_Nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_Nodes.append(gNode)
        judge0 = [node.op for node in compute_Nodes] in [
            ["MatMul", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Mul", "MatMul"],
            ["MatMul", "Add", "Reshape", "Transpose", "Reshape", "Transpose", "MatMul"],
        ]

        judge1 = None
        shape0 = None
        shape1 = None
        if judge0:
            for node in compute_Nodes:
                if node.op == "Reshape":
                    shape0 = node.outputs[0].shape
            shape1 = compute_Nodes[-2].outputs[0].shape
            judge1 = (shape0[-1] == shape1[-2])
        
        return judge0 and judge1
    
    def pass_6_7(path):
        res = {"in_project_weight_k": getConstInput(path[0])[-1]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["in_project_bias_k"] = getConstInput(gNode)[0]
                break
        return res
    
    # 5->6
    def judge_7_0(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for i, gNode in enumerate(paths[0]):
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["MatMul", "Reshape", "Add", "Reshape", "Softmax"],
            ["MatMul", "Div", "Add", "Softmax"],
            ["MatMul", "Mul", "Add", "Softmax"],
            ["MatMul", "Softmax"]
        ]
        return judge0
    
    def pass_7_0(path):
        res = {}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                index = 0 if gNode.inputs[1] == path[i - 1] else 1
                res["key_padding_mask"] = gNode.inputs[index]
                break
        return res

    # 6->7
    def judge_0_1(paths):
        if len(paths) != 1:
            return False
        compute_Nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast" and gNode.op != "Dropout":
                compute_Nodes.append(gNode)
        judge0 = [node.op for node in compute_Nodes] in [
            ["Softmax", "MatMul"],
            ["Softmax", "Dropout", "MatMul"]
        ]
        return judge0
    
    # 7->8
    def judge_1_2(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["MatMul", "Transpose", "Reshape", "MatMul"],
            ["MatMul", "Reshape", "Transpose", "Reshape", "MatMul"]
        ]
        return judge0

    # 8->9
    def judge_2_3(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = True
        if [node.op for node in compute_nodes] == ["MatMul", "Add"]:
            judge0 = getConstInput(compute_nodes[0])[0].shape[-1] == \
                getConstInput(compute_nodes[1])[0].shape[-1]
        elif [node.op for node in compute_nodes] == ["MatMul"]:
            if len(paths[0][-1].outputs) == 1 and paths[0][-1].outputs[0].op == "Add":
                add = paths[0][-1].outputs[0]
                if getConstInput(compute_nodes[0])[0].shape[-1] == \
                    getConstInput(add)[0].shape[-1]:
                    judge0 = False
        else:
            judge0 = False
        return judge0
    
    def pass_2_3(path):
        res = {"out_project_weight": getConstInput(path[0])[0]}
        for i, gNode in enumerate(path):
            if isNode(gNode) and gNode.op == "Add":
                if len(getConstInput(gNode)) == 1:
                    res["out_project_bias"] = getConstInput(gNode)[0]
                break
        res["output"] = path[-1]
        return res

    judges = [
        [judge_0_1],
        [judge_1_2],
        [judge_2_3],
        [],
        [judge_4_1],
        [judge_5_7],
        [judge_6_7],
        [judge_7_0],
        [judge_8_46, judge_8_46],
        [judge_9_5]
    ]
    passes = [
        [None],
        [None],
        [pass_2_3],
        [],
        [pass_4_1],
        [pass_5_7],
        [pass_6_7],
        [pass_7_0],
        [pass_8_4, None],
        [pass_9_5]
    ]
    pattern_detail_list = list()
    print("patterns: {}".format(len(patterns)))
    for pattern in tqdm(patterns):
        label = True
        pattern_detail = copy.deepcopy(edges)
        for start in range(len(edges)):
            if not label:
                break
            for i, end in enumerate(edges[start]):
                if not label:
                    break
                paths = traversalPatternMatching(
                    graph,
                    lambda x: x.name == pattern[start].name,
                    lambda x, path: not x.shape_or_index,
                    lambda x, path: x.name == pattern[end].name,
                    max_depth=max_depth
                )
                label = label and judges[start][i](paths)
                pattern_detail[start][i] = paths[0]
        if label:
            pattern_detail_list.append(pattern_detail)
    
    # res = list()
    count = 0
    for pattern_detail in pattern_detail_list:
        args_relay = {} 
        for start in range(len(edges)):
            for i, end in enumerate(edges[start]):
                if not passes[start][i] is None:
                    args_relay.update(passes[start][i](pattern_detail[start][i]))
        new_name = f"{op_name}_{count}"
        args_final = args_process(args_relay, new_name)

        if args_final["input"].dtype != args_final["output"].dtype:
            continue

        SelfAttention_node = gs.Node(
            op=op_name,
            name=new_name,
            attrs={
                "num_heads": args_final["num_heads"], 
                "l2norm": args_final["l2norm"], 
                "attn_type": args_final["attn_type"]
            },
            inputs=[
                args_final["input"], 
                args_final["context"], 
                args_final["key_padding_mask"], 
                args_final["in_project_weight_q"], 
                args_final["in_project_weight_k"], 
                args_final["in_project_weight_v"], 
                args_final["out_project_weight"], 
                args_final["out_project_bias"], 
                args_final["gamma"], 
                args_final["temperature"]
            ],
            outputs=[args_final["output"]],
        )
        SelfAttention_node.domain = "pmx"
        graph.nodes.append(SelfAttention_node)
        count += 1
        input_name = args_final["input"].name
        context_name = args_final["context"].name
        output_name = args_final["output"].name
        print(f"[I]    find {new_name} between {input_name}, {context_name} -> {output_name}")
        # res.append((pattern_detail, args_final))
        pattern_detail[output_edge][0][-2].outputs = []
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_GroupNorm(graph, max_depth=30):
    op_name = "GroupNorm"
    print(f"[I] Merging {op_name}")
    setShapeOrIndex(graph)
    gNode_feature = [
        lambda x: isNode(x) and x.op == "InstanceNormalization" and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "Reshape" and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "Mul" and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "Add" and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isVariable(x) and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isVariable(x) and hasattr(x, "shape_or_index") and not x.shape_or_index,
        lambda x: isNode(x) and x.op == "Reshape" and hasattr(x, "shape_or_index") and not x.shape_or_index and \
            len(getConstInput(x)) == 1,
    ]
    edges = [
        [1],
        [2],
        [3],
        [4],
        [],
        [6],
        [0],
    ]
    output_edge = 3
    patterns = subGraphPatternMatching(
        graph,
        gNode_feature = gNode_feature,
        edges = edges,
        max_depth=max_depth
    )

    def args_process(args_relay, new_name):
        res = {
            "eps": 0,
            "group": None,
            "moving_average_fraction": 0.999,
            "use_global_stats": 1,
            "input": None,
            "weight_constant": None,
            "beta_constant": None,
            "output": None
        }
        dtype = args_relay["weight_constant"].values.dtype

        for key in res.keys():
            if key in args_relay.keys():
                res[key] = args_relay[key]
            elif res[key] is None:
                res[key] = gs.Constant(
                    f"{new_name}/{key}", 
                    values=np.array([], dtype=dtype)
                )

        return res

    # 0->1
    def judge_0_1(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["InstanceNormalization", "Reshape"],
        ]
        return judge0
    
    # 1->2
    def judge_1_2(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["Reshape", "Mul"],
        ]
        return judge0

    # 2->3
    def judge_2_3(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["Mul", "Add"],
        ]
        return judge0

    def pass_2_3(path):
        res = {
            "weight_constant": getConstInput(path[0])[0],
            "beta_constant": getConstInput(path[-1])[0],
        }
        return res
    
    # 3->4
    def judge_3_4(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode):
                if gNode.op == "Cast":
                    return False
                else:
                    compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["Add"],
        ]
        return judge0

    def pass_3_4(path):
        return {"output": path[-1]}
    
    # 5->6
    def judge_5_6(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode):
                if gNode.op == "Cast":
                    return False
                else:
                    compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["Reshape"],
        ]
        return judge0
    
    def pass_5_6(path):
        shape = getConstInput(path[-1])[0]
        return {
            "input": path[0], 
            "group": shape.values[-2]
        }
    
    # 6->0
    def judge_6_0(paths):
        if len(paths) != 1:
            return False
        compute_nodes = list()
        for gNode in paths[0]:
            if isNode(gNode) and gNode.op != "Cast":
                compute_nodes.append(gNode)
        judge0 = [node.op for node in compute_nodes] in [
            ["Reshape", "InstanceNormalization"],
        ]
        return judge0
    
    def pass_6_0(path):
        return {"eps": path[-1].attrs['epsilon']}

    judges = [
        [judge_0_1],
        [judge_1_2],
        [judge_2_3],
        [judge_3_4],
        [],
        [judge_5_6],
        [judge_6_0],
    ]
    passes = [
        [None],
        [None],
        [pass_2_3],
        [pass_3_4],
        [],
        [pass_5_6],
        [pass_6_0],
    ]
    pattern_detail_list = list()
    for pattern in tqdm(patterns):
        label = True
        pattern_detail = copy.deepcopy(edges)
        for start in range(len(edges)):
            if not label:
                break
            for i, end in enumerate(edges[start]):
                if not label:
                    break
                paths = traversalPatternMatching(
                    graph,
                    lambda x: x.name == pattern[start].name,
                    lambda x, path: not x.shape_or_index,
                    lambda x, path: x.name == pattern[end].name,
                    max_depth=max_depth
                )
                label = label and judges[start][i](paths)
                pattern_detail[start][i] = paths[0]
        if label:
            pattern_detail_list.append(pattern_detail)
    
    # res = list()
    count = 0
    for pattern_detail in pattern_detail_list:
        args_relay = {} 
        for start in range(len(edges)):
            for i, end in enumerate(edges[start]):
                if not passes[start][i] is None:
                    args_relay.update(passes[start][i](pattern_detail[start][i]))
        new_name = f"{op_name}_{count}"
        args_final = args_process(args_relay, new_name)

        SelfAttention_node = gs.Node(
            op=op_name,
            name=new_name,
            attrs={
                "eps": args_final["eps"], 
                "group": args_final["group"], 
                "moving_average_fraction": args_final["moving_average_fraction"],
                "use_global_stats": args_final["use_global_stats"]
            },
            inputs=[
                args_final["input"], 
                args_final["weight_constant"], 
                args_final["beta_constant"], 
            ],
            outputs=[args_final["output"]],
        )
        SelfAttention_node.domain = "pmx"
        graph.nodes.append(SelfAttention_node)
        count += 1
        input_name = args_final["input"].name
        output_name = args_final["output"].name
        print(f"[I]    find {new_name} between {input_name} -> {output_name}")
        # res.append((pattern_detail, args_final))
        pattern_detail[output_edge][0][-2].outputs = []
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_SiLU(graph):
    op_name = "SiLU"
    print(f"[I] Merging {op_name}")
    count = 0
    for sigmoid_node in graph.nodes:
        if sigmoid_node.op == "Sigmoid":
            if len(sigmoid_node.outputs[0].outputs) != 1:
                continue
            mul = sigmoid_node.outputs[0].outputs[0]
            if mul.op != "Mul":
                continue
            cur_in = None
            for mul_in in mul.inputs:
                if mul_in.name != sigmoid_node.outputs[0].name:
                    cur_in = mul_in

            cur_out = mul.outputs[0]
            mul.outputs = []
            new_name = f"{op_name}_{count}"
            input_name = cur_in.name
            output_name = cur_out.name
            SiLU_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[cur_in], 
                outputs=[cur_out]
            )
            SiLU_node.domain = "pmx"
            graph.nodes.append(SiLU_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_GroupNormMul(graph):
    op_name = "GroupNormMul"
    print(f"[I] Merging {op_name}")
    count = 0
    for silu in graph.nodes:
        if silu.op == "SiLU":
            if len(silu.inputs[0].inputs) != 1 or silu.inputs[0].inputs[0].op != "GroupNorm":
                continue
            gn_node = silu.inputs[0].inputs[0]
            
            cast0 = None
            cast1 = None
            if len(gn_node.inputs[0].inputs) == 1 and gn_node.inputs[0].inputs[0].op == "Cast":
                cast0 = gn_node.inputs[0].inputs[0]
            if len(silu.outputs[0].outputs) == 1 and silu.outputs[0].outputs[0].op == "Cast":
                cast1 = silu.outputs[0].outputs[0]
            
            input = gn_node.inputs[0] if (cast0 is None or cast1 is None) else cast0.inputs[0]
            weight_constant = gn_node.inputs[1]
            beta_constant = gn_node.inputs[2]
            cur_out = silu.outputs[0] if (cast0 is None or cast1 is None) else cast1.outputs[0]
            last_node = silu if (cast0 is None or cast1 is None) else cast1
            last_node.outputs = []

            new_name = f"{op_name}_{count}"
            input_name = input.name
            weight_constant_name = weight_constant.name
            beta_constant_name = beta_constant.name
            output_name = cur_out.name
            GroupNormMul_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[input, weight_constant, beta_constant], 
                outputs=[cur_out], 
                attrs=gn_node.attrs
            )
            gn_node.inputs = []
            GroupNormMul_node.domain = "pmx"
            graph.nodes.append(GroupNormMul_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name}, {weight_constant_name}, {beta_constant_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_AddGroupNormMul(graph):
    op_name = "AddGroupNormMul"
    print(f"[I] Merging {op_name}")
    count = 0
    for group_norm_mul in graph.nodes:
        if group_norm_mul.op == "GroupNormMul":
            add_node = None
            if not(len(group_norm_mul.inputs[0].inputs) == 1 and group_norm_mul.inputs[0].inputs[0].op == "Add"):
                continue
            add_node = group_norm_mul.inputs[0].inputs[0]
            
            if len(add_node.outputs[0].outputs) != 1:
                continue
            
            index = None
            for i in range(len(add_node.inputs)):
                if len(add_node.inputs[i].inputs) == 1 and add_node.inputs[i].inputs[0].op == "Unsqueeze":
                    index = i
                    break
            unsqueeze1 = None
            unsqueeze0 = None
            if index is None:
                continue
            unsqueeze1 = add_node.inputs[index].inputs[0]
            if not(len(unsqueeze1.inputs[0].inputs) == 1 and unsqueeze1.inputs[0].inputs[0].op == "Unsqueeze"):
                continue
            unsqueeze0 = unsqueeze1.inputs[0].inputs[0]

            input_a = add_node.inputs[0] if index == 1 else add_node.inputs[1]
            scale = group_norm_mul.inputs[1]
            shift = group_norm_mul.inputs[2]
            input_b = unsqueeze0.inputs[0]
            cur_out = group_norm_mul.outputs[0]
            
            new_name = f"{op_name}_{count}"
            input_a_name = input_a.name
            scale_name = scale.name
            shift_name = shift.name
            input_b_name = input_b.name
            output_name = cur_out.name
            gn_sig_mul_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[input_a, scale, shift, input_b], 
                outputs=[cur_out], 
                attrs=group_norm_mul.attrs)
            gn_sig_mul_node.domain = "pmx"
            graph.nodes.append(gn_sig_mul_node)
            unsqueeze0.inputs = []
            add_node.inputs = []
            group_norm_mul.outputs = []
            count += 1
            print(f"[I]    find {new_name} between {input_a_name}, {scale_name}, {shift_name}, {input_b_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_LayerNorm(graph):
    op_name = "LayerNorm"
    count = 0
    for layerNormalization in graph.nodes:
        if layerNormalization.op == "LayerNormalization":
            cur_out = layerNormalization.outputs[0]
            cur_in = layerNormalization.inputs[0]
            gamma_constant = layerNormalization.inputs[1]
            beta_constant = layerNormalization.inputs[2]
            axis = layerNormalization.attrs["axis"]
            eps_value = layerNormalization.attrs["epsilon"]
            layerNormalization.inputs = []
            layerNormalization.outputs = []

            new_name = f"{op_name}_{count}"
            input_name = cur_in.name
            output_name = cur_out.name
            LayerNorm_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[cur_in, gamma_constant, beta_constant], 
                outputs=[cur_out], 
                attrs={'axis':axis, 'eps': eps_value, 'elementwise_affine': 1}
            )
            LayerNorm_node.domain = "pmx"
            graph.nodes.append(LayerNorm_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name} -> {output_name}")

    for add1 in graph.nodes:
        if add1.op == "Add":
            possible_mul = searchgNodeFromParents(
                add1,
                lambda x: isNode(x) and x.op == "Mul",
                max_depth=4,
                include_root=False
            )
            # possible_mul = search_nodes_from_parent(graph, add1, "Mul", 2)
            if len(possible_mul) == 0:
                continue
            mul = possible_mul[0][0]

            possible_div = searchgNodeFromParents(
                mul,
                lambda x: isNode(x) and x.op == "Div",
                max_depth=4,
                include_root=False
            )
            # possible_div = search_nodes_from_parent(graph, mul, "Div", 2)
            if len(possible_div) == 0:
                continue
            div = possible_div[0][0]

            possible_sqrt = searchgNodeFromParents(
                div,
                lambda x: isNode(x) and x.op == "Sqrt",
                max_depth=4,
                include_root=False
            )
            # possible_sqrt = search_nodes_from_parent(graph, div, "Sqrt", 2)
            if len(possible_sqrt) == 0:
                continue
            sqrt = possible_sqrt[0][0]

            possible_add0 = searchgNodeFromParents(
                sqrt,
                lambda x: isNode(x) and x.op == "Add",
                max_depth=4,
                include_root=False
            )
            # possible_add0 = search_nodes_from_parent(graph, sqrt, "Add", 2)
            if len(possible_add0) == 0:
                continue
            add0 = possible_add0[0][0]

            eps_value = 10e-5
            if isinstance(add0.inputs[1], gs.ir.tensor.Variable):
                assert add0.inputs[1].inputs[0].op == 'Constant'
                eps_value = float(add0.inputs[1].inputs[0].attrs['value'].values)
            elif isinstance(add0.inputs[1], gs.ir.tensor.Constant):
                assert isinstance(add0.inputs[1], gs.ir.tensor.Constant)
                eps_value = float(add0.inputs[1].values)

            possible_reducemean1 = searchgNodeFromParents(
                add0,
                lambda x: isNode(x) and x.op == "ReduceMean",
                max_depth=4,
                include_root=False
            )
            # possible_reducemean1 = search_nodes_from_parent(graph, add0, "ReduceMean", 2)
            if len(possible_reducemean1) == 0:
                continue
            reducemean1 = possible_reducemean1[0][0]
            
            possible_pow = searchgNodeFromParents(
                reducemean1,
                lambda x: isNode(x) and x.op == "Pow",
                max_depth=4,
                include_root=False
            )
            # possible_pow = search_nodes_from_parent(graph, reducemean1, "Pow", 2)
            if len(possible_pow) == 0:
                continue
            pow = possible_pow[0][0]
            
            possible_sub = searchgNodeFromParents(
                pow,
                lambda x: isNode(x) and x.op == "Sub",
                max_depth=4,
                include_root=False
            )
            # possible_sub = search_nodes_from_parent(graph, pow, "Sub", 2)
            if len(possible_sub) == 0:
                continue
            sub = possible_sub[0][0]
            
            possible_reducemean0 = searchgNodeFromParents(
                sub,
                lambda x: isNode(x) and x.op == "ReduceMean",
                max_depth=4,
                include_root=False
            )
            # possible_reducemean0 = search_nodes_from_parent(graph, sub, "ReduceMean", 2)
            if len(possible_reducemean0) == 0:
                continue
            reducemean0 = possible_reducemean0[0][0]
            
            if pow.inputs[0] != div.inputs[0]:
                continue
            if reducemean0.inputs[0] != sub.inputs[0]:
                continue

            axes = reducemean0.attrs['axes']
            assert isinstance(axes, list)
            axis = axes[0]

            if isinstance(mul.inputs[1], gs.ir.tensor.Constant):
                gamma_constant = mul.inputs[1]
            else:
                gamma_constant = mul.inputs[1].inputs[0].inputs[0]

            if isinstance(add1.inputs[1], gs.ir.tensor.Constant):
                beta_constant = add1.inputs[1]
            else:
                beta_constant = add1.inputs[1].inputs[0].inputs[0]

            cur_out = add1.outputs[0]
            cur_in = reducemean0.inputs[0]
            reducemean0.inputs = []
            add1.outputs = []

            new_name = f"{op_name}_{count}"
            input_name = cur_in.name
            output_name = cur_out.name
            LayerNorm_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[cur_in, gamma_constant, beta_constant], 
                outputs=[cur_out], 
                attrs={'axis':axis, 'eps': eps_value, 'elementwise_affine': 1}
            )
            LayerNorm_node.domain = "pmx"
            graph.nodes.append(LayerNorm_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_gegelu(graph):
    op_name = "GEGELU"
    count = 0
    for mul2 in graph.nodes:
        if mul2.op == "Mul":
            possible_slice0 = searchgNodeFromParents(
                mul2,
                lambda x: isNode(x) and x.op == "Slice",
                max_depth=4,
                include_root=False
            )
            if len(possible_slice0) == 0:
                continue
            slice0 = possible_slice0[0][0]

            # find both "Slice" nodes
            possible_slice1 = searchgNodeFromParents(
                mul2,
                lambda x: isNode(x) and x.op == "Slice",
                max_depth=10,
                include_root=False
            )
            slice1 = None
            if len(possible_slice1) < 2:
                continue
            for slice in possible_slice1:
                slice1 = slice[0]
                if slice1 != slice0:
                    break
                
            # as the defination in onnx, the first input of slice is data
            if slice0.inputs[0] != slice1.inputs[0]:
                continue

            possible_div = searchgNodeFromChildren(
                slice1,
                lambda x: isNode(x) and x.op == "Div",
                max_depth=4,
                include_root=False
            )
            if len(possible_div) == 0:
                continue
            div = possible_div[0][0]

            possible_erf = searchgNodeFromChildren(
                div,
                lambda x: isNode(x) and x.op == "Erf",
                max_depth=4,
                include_root=False
            )
            if len(possible_erf) == 0:
                continue
            erf = possible_erf[0][0]

            possible_add1 = searchgNodeFromChildren(
                erf,
                lambda x: isNode(x) and x.op == "Add",
                max_depth=4,
                include_root=False
            )
            if len(possible_add1) == 0:
                continue
            add1 = possible_add1[0][0]

            possible_mul = searchgNodeFromChildren(
                add1,
                lambda x: isNode(x) and x.op == "Mul",
                max_depth=10,
                include_root=False
            )
            if len(possible_mul) < 3:
                continue
            if possible_mul[2][0] != mul2:
                continue
            
            cur_in = slice0.inputs[0]
            cur_out = mul2.outputs[0]

            slice0.inputs = []
            slice1.inputs = []
            mul2.outputs = []

            new_name = f"{op_name}_{count}"
            input_name = cur_in.name
            output_name = cur_out.name
            gelu_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[cur_in], 
                outputs=[cur_out]
            )
            gelu_node.domain = "pmx"
            graph.nodes.append(gelu_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_ReshapeSqueeze(graph):
    '''
    N, C, H, W -> N, H*W, C
    Make sure that shape inference has been done
    '''
    op_name = "ReshapeSqueeze"
    count = 0
    for reshape in graph.nodes:
        if reshape.op == "Reshape":
            if not(len(reshape.inputs[0].inputs) == 1 and reshape.inputs[0].inputs[0].op == "Transpose"):
                continue
            # if len(reshape.inputs[1].inputs) == 0:
            #     continue
            data_node = reshape.inputs[0].inputs[0]
            # shape_node = reshape.inputs[1].inputs[0]
            
            in_shape = data_node.inputs[0].shape    # N, C, H, W
            out_shape = reshape.outputs[0].shape    # N, H*W, C
            if not(f"{in_shape[-2]}*{in_shape[-1]}" == out_shape[-2] or \
            f"({in_shape[-2]})*({in_shape[-1]})" == out_shape[-2] or \
            in_shape[-2]*in_shape[-1] == out_shape[-2]):
                continue
            cur_out = reshape.outputs[0]
            cur_in = data_node.inputs[0]
            data_node.inputs = []
            reshape.outputs = []

            new_name = f"{op_name}_{count}"
            input_name = cur_in.name
            output_name = cur_out.name
            reshapes_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[cur_in], 
                outputs=[cur_out])
            reshapes_node.domain = "pmx"
            graph.nodes.append(reshapes_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def merge_ReshapeUnsqueeze(graph):
    '''
    N, H*W, C -> N, C, H, W
    Make sure that shape inference has been done
    '''
    op_name = "ReshapeUnsqueeze"
    count = 0
    for transpose in graph.nodes:
        if transpose.op == "Transpose":
            if not(len(transpose.inputs[0].inputs) == 1 and transpose.inputs[0].inputs[0].op == "Reshape"):
                continue
            reshape = transpose.inputs[0].inputs[0]

            # if len(reshape.inputs[0].inputs) == 0 or len(reshape.inputs[1].inputs) == 0:
            #     continue
            in_shape = reshape.inputs[0].shape    # N, H*W, C
            out_shape = transpose.outputs[0].shape    # N, C, H, W
            if not(f"{out_shape[-2]}*{out_shape[-1]}" == in_shape[-2] or \
            f"({out_shape[-2]})*({out_shape[-1]})" == in_shape[-2] or \
            out_shape[-2]*out_shape[-1] == in_shape[-2]):
                continue
            cur_out = transpose.outputs[0]
            cur_in = reshape.inputs[0]
            reshape.inputs = []
            transpose.outputs = []

            new_name = f"{op_name}_{count}"
            input_name = cur_in.name
            output_name = cur_out.name
            reshapes_node = gs.Node(
                op=op_name, 
                name=new_name,
                inputs=[cur_in], 
                outputs=[cur_out]
            )
            reshapes_node.domain = "pmx"
            graph.nodes.append(reshapes_node)
            count += 1
            print(f"[I]    find {new_name} between {input_name} -> {output_name}")
    print(f"[I] Merge {op_name} count: {count}")
    graph.cleanup().toposort()

def erase_div_1(graph):
    for div in graph.nodes:
        if div.op == "Div":
            constant_inputs = getConstInput(div)
            if len(constant_inputs) == 1 and constant_inputs[0].values.size == 1 and \
                float(constant_inputs[0].values) - 1 < 1e-3:
                index = 0 if constant_inputs[0] == div.inputs[1] else 1
                identity = gs.Node(
                    op="Identity", 
                    name=f"{div.name}_Identity", 
                    inputs=[div.inputs[index]], 
                    outputs=div.outputs
                )
                graph.nodes.append(identity)
                div.inputs = []
                div.outputs = []
    graph.cleanup().toposort()
