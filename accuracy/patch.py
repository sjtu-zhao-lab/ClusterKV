import types
from transformers.models.llama.modeling_llama import LlamaAttention
from accuracy.quest_attention import forward_quest, forward_quest_glm
from accuracy.cluster_attention import forward_cluster, forward_cluster_glm, apply_cluster_config

global layer_id
layer_id = 32

def parse_common_args(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama3-8b-chat-4k",
            "llama3-8b-chat-8k",
            "llama3.1-8b-chat-4k",
            "llama3.1-8b-chat-8k",
            "llama3.1-8b-chat-32k",
            "glm4-9b-chat-4k",
            "glm4-9b-chat-8k",
            "glm4-9b-chat-32k",
        ],
    )
    parser.add_argument("--token_budget", type=int, default=1024)

    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")

    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--cluster", action="store_true", help="Enable ClusterKV Attention")
    parser.add_argument("--head_sel", type=str, choices=["truc", "pad"], default="truc",
                        help="truncate or pad for different heads to make same selection budget")
    parser.add_argument("--balance", action="store_true", help="Use Balanced KMeans")
    parser.add_argument("--nlist", type=int, default=400, help="Number of clusters")
    parser.add_argument("--fit_iter", type=int, default=20, help="Max steps for clustering")
    parser.add_argument("--gqa_policy", type=str, choices=["qavg"], default=None)
    parser.add_argument("--dist_t", type=str, 
                        choices=["cosine", "inner_product", "l2", "l1", "euclidean", 
                                 "chebyshev", "canberra"], 
                        default="cosine", help="Distance for clustering")

    parser.add_argument("--cache_steps", type=int, default=0, 
                        help="Stat cache hit rate of recent steps")
    parser.add_argument("--topk_stat", action="store_true", help="Stat hit rate of TopK tokens")
    return parser


def enable_attention_eval(model_name, model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_attention_eval(model_name, module, args)

        global layer_id
        if isinstance(module, LlamaAttention):
            layer_id -= 1
            module.layer_id = layer_id
            module.flash_forward = module.forward
            module.cache_steps = args.cache_steps
            module.token_budget = args.token_budget
            module.chunk_size = args.chunk_size
            module.cluster_cache = None
            if args.quest:
                module.forward = types.MethodType(forward_quest, module)
                module.gen = False
            elif args.cluster:
                module.forward = types.MethodType(forward_cluster, module)
                apply_cluster_config(module, args)
            module.topk_stat = True if args.topk_stat else False

        elif "glm4" in model_name and module.__class__.__name__ == "SelfAttention":
            module.flash_forward = module.forward
            module.cache_steps = args.cache_steps
            module.token_budget = args.token_budget
            module.chunk_size = args.chunk_size
            module.cluster_cache = None
            if args.quest:
                module.forward = types.MethodType(forward_quest_glm, module)
                module.gen = False
            elif args.cluster:
                module.forward = types.MethodType(forward_cluster_glm, module)
                apply_cluster_config(module, args)
            module.topk_stat = True if args.topk_stat else False


def get_config_output_affix(args):
    config_affix = ''
    if args.quest:
        if args.chunk_size == 16:
            config_affix = f"-{args.token_budget}"
        else:
            config_affix = f"-{args.token_budget}c{args.chunk_size}"
    elif args.cluster:
        km_id = ("h" + args.head_sel + ("b" if args.balance else "") 
                 + str(args.nlist) + f"fi{args.fit_iter}")
        if args.sink > 0:
            km_id += f"sink{args.sink}"
        config_affix = f"-{km_id}_{args.token_budget}"
    return config_affix