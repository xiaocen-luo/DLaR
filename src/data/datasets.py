from datasets import load_dataset, Dataset, concatenate_datasets, get_dataset_config_names
import os
import json

SHOTS = [
    [
        {'role': 'user', 'content': "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction."},
        {'role': 'assistant', 'content': "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}}"}
    ],
    [
        {'role': 'user', 'content': "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?"},
        {'role': 'assistant', 'content': "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi"}
    ],
    [
        {'role': 'user', 'content': "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?"},
        {'role': 'assistant', 'content': "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}"}
    ],
    [
        {'role': 'user', 'content': "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?"},
        {'role': 'assistant', 'content': "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0"}
    ]
]

SYSTEM_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}."""
SYSTEM_PROMPT2 = """Please reason step by step, and put your final answer within \\boxed{} with (A, B, C or D)."""

# 本地数据路径配置
LOCAL_DATA_PATHS = {
    "gsm8k": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/gsm8k",
    "math500": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/MATH-500",
    "hendrycks_math": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/hendrycks_math",
    "aime2024": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/aime_2024",
    "aime2025": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/aime_2025",
    "minervamath": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/minervamath",
    "amc23": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/amc23",
    "arc_challenge": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/arc_challenge",
    "mmlu_stem": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/mmlu_stem",
    "gpqa_diamond": "/public/home/xiangyuduan/xcluo/lepo-main/datasets/gpqa_diamond",  # 新增
}

def get_prompt_shot(question, shot=0):
    prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for i in range(shot):
        prompt.extend(SHOTS[i])
    prompt.extend([{'role': 'user', 'content': question + '\n' + SYSTEM_PROMPT}])
    return prompt

def load_local_dataset(dataset_name, split='train', subset=None, file_format='parquet'):
    """通用本地数据集加载函数"""
    import glob
    
    base_path = LOCAL_DATA_PATHS.get(dataset_name)
    if base_path is None:
        raise ValueError(f"No local path configured for dataset: {dataset_name}")
    
    # 构建数据集路径
    if subset:
        # 检查 subset 是否是文件名（包含扩展名）
        if subset.endswith('.jsonl') or subset.endswith('.json') or subset.endswith('.parquet'):
            dataset_path = base_path
            specific_file = subset
        else:
            dataset_path = os.path.join(base_path, subset)
            specific_file = None
    else:
        dataset_path = base_path
        specific_file = None
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Local dataset not found at: {dataset_path}")
    
    # 尝试加载不同格式的数据
    try:
        # 情况1: 指定了具体文件
        if specific_file:
            file_path = os.path.join(dataset_path, specific_file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            print(f"Loading specific file: {file_path}")
            if specific_file.endswith('.jsonl'):
                dataset = load_dataset('json', data_files=file_path, split='train')
                return dataset
            elif specific_file.endswith('.json'):
                dataset = load_dataset('json', data_files=file_path, split='train')
                return dataset
            elif specific_file.endswith('.parquet'):
                dataset = load_dataset('parquet', data_files=file_path, split='train')
                return dataset
        
        # 情况2: 尝试作为 HuggingFace 格式加载
        if os.path.exists(os.path.join(dataset_path, 'dataset_info.json')):
            return load_dataset(dataset_path, split=split)
        
        # 情况3: 尝试加载 parquet 文件
        parquet_files = glob.glob(os.path.join(dataset_path, f"{split}*.parquet"))
        if not parquet_files:
            parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
        
        if parquet_files:
            print(f"Loading parquet files: {parquet_files}")
            if len(parquet_files) == 1:
                return load_dataset('parquet', data_files=parquet_files[0], split='train')
            else:
                return load_dataset('parquet', data_files=parquet_files, split='train')
        
        # 情况4: 尝试加载 jsonl 文件
        # 首先尝试匹配 split 名称的文件
        jsonl_files = glob.glob(os.path.join(dataset_path, f"{split}.jsonl"))
        if not jsonl_files:
            jsonl_files = glob.glob(os.path.join(dataset_path, f"{split}*.jsonl"))
        if not jsonl_files:
            jsonl_files = glob.glob(os.path.join(dataset_path, f"*{split}*.jsonl"))
        if not jsonl_files:
            jsonl_files = glob.glob(os.path.join(dataset_path, "*.jsonl"))
        
        if jsonl_files:
            print(f"Loading jsonl files: {jsonl_files}")
            if len(jsonl_files) == 1:
                dataset = load_dataset('json', data_files=jsonl_files[0], split='train')
                return dataset
            else:
                # 多个文件，加载所有
                dataset = load_dataset('json', data_files=jsonl_files, split='train')
                return dataset
        
        # 情况5: 尝试加载 json 文件
        json_files = glob.glob(os.path.join(dataset_path, f"{split}.json"))
        if not json_files:
            json_files = glob.glob(os.path.join(dataset_path, f"{split}*.json"))
        if not json_files:
            json_files = glob.glob(os.path.join(dataset_path, f"*{split}*.json"))
        if not json_files:
            json_files = glob.glob(os.path.join(dataset_path, "*.json"))
        
        if json_files:
            print(f"Loading json files: {json_files}")
            if len(json_files) == 1:
                dataset = load_dataset('json', data_files=json_files[0], split='train')
                return dataset
            else:
                dataset = load_dataset('json', data_files=json_files, split='train')
                return dataset
        
        raise ValueError(f"No supported data files found in {dataset_path}")
        
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        raise

def get_gsm8k_questions(split="train", shot=0, **kwargs) -> Dataset:
    def extract_hash_answer(text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")
    
    try:
        data = load_local_dataset("gsm8k", split=split)
        
        # 标准化列名
        if "problem" in data.column_names:
            print(f"Renaming 'problem' to 'question' in GSM8K dataset")
            data = data.rename_column("problem", "question")
        
        if "answer" not in data.column_names and "solution" in data.column_names:
            print(f"Using 'solution' as 'answer' in GSM8K dataset")
            data = data.rename_column("solution", "answer")
        
        # 打印调试信息
        print(f"GSM8K dataset columns after standardization: {data.column_names}")
        
    except Exception as e:
        print(f"Warning: Local GSM8K dataset not found or invalid: {e}")
        print("Using empty dataset")
        data = Dataset.from_dict({'question': [], 'answer': []})
    
    # 应用 prompt 和 solution 格式化
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': f"{extract_hash_answer(x['answer'])}" if 'answer' in x else "",
        'question': x['question']
    })
    return data

def get_math500_questions(shot=0, **kwargs) -> Dataset:
    try:
        # 直接加载 test.jsonl 文件
        dataset = load_local_dataset("math500", split='test', subset='test.jsonl')
        print(f"Successfully loaded MATH-500 dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Warning: Local MATH-500 dataset not found, using empty dataset. Error: {e}")
        dataset = Dataset.from_dict({'problem': [], 'solution': []})
    
    # 检查数据集的列名
    print(f"Dataset columns: {dataset.column_names}")
    
    # 根据实际的列名进行映射
    dataset = dataset.map(lambda x: {
        'prompt': get_prompt_shot(x.get('problem', x.get('question', '')), shot),
        'solution': x.get('solution', x.get('answer', '')),
        'question': x.get('problem', x.get('question', ''))
    })
    return dataset

def get_math_questions(split="train", subset='algebra', shot=0, **kwargs) -> Dataset:
    if subset == 'all':
        name_list = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    elif isinstance(subset, str):
        name_list = [subset]
    elif isinstance(subset, list):
        name_list = subset
    else:
        raise ValueError('subset must be a string or a list of strings')
    
    data = Dataset.from_dict({})
    for name in name_list:
        try:
            data0 = load_local_dataset("hendrycks_math", split=split, subset=name)
            data = concatenate_datasets([data, data0])
        except:
            print(f"Warning: Local math dataset subset '{name}' not found, skipping")
    
    if len(data) == 0:
        print("Warning: No math dataset subsets found, using empty dataset")
        return Dataset.from_dict({'problem': [], 'solution': []})
    
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['problem'], shot),
        'solution': x['solution'],
        'question': x['problem']
    })
    return data

def get_dapo_questions(split="train", subset='all', shot=0, **kwargs) -> Dataset:
    base_path = "/public/home/xiangyuduan/xcluo/lepo-main/datasets/DAPO-Math-17k-Processed"
    dataset_path = os.path.join(base_path, subset)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"DAPO dataset not found at: {dataset_path}")
    
    # 检查文件格式
    if any(f.endswith('.parquet') for f in os.listdir(dataset_path)):
        data = load_dataset('parquet', data_files=os.path.join(dataset_path, '*.parquet'), split='train')
    elif any(f.endswith('.jsonl') for f in os.listdir(dataset_path)):
        data = load_dataset('json', data_files=os.path.join(dataset_path, '*.jsonl'), split='train')
    else:
        data = load_dataset(dataset_path, split='train')
    
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['prompt'], shot),
        'solution': x['solution'],
        'question': x['prompt']
    })
    
    train_test_split = data.train_test_split(test_size=0.1, seed=42)
    return train_test_split[split]

def get_aime2024_questions(shot=0, **kwargs) -> Dataset:
    try:
        data = load_local_dataset("aime2024", split='train')
    except:
        print("Warning: Local AIME2024 dataset not found, using empty dataset")
        data = Dataset.from_dict({'problem': [], 'answer': []})
    
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['problem'], shot),
        'solution': x['answer'],
        'question': x['problem']
    })
    return data

def get_aime2025_questions(shot=0, **kwargs) -> Dataset:
    try:
        # 直接加载具体的 jsonl 文件
        data1 = load_local_dataset("aime2025", split='train', subset='AIME2025-I.jsonl')
        data2 = load_local_dataset("aime2025", split='train', subset='AIME2025-II.jsonl')
        data = concatenate_datasets([data1, data2])
        print(f"Successfully loaded AIME2025 dataset with {len(data)} examples")
    except Exception as e:
        print(f"Warning: Local AIME2025 dataset not found, using empty dataset. Error: {e}")
        data = Dataset.from_dict({'question': [], 'answer': []})
    
    # 检查数据集的列名
    print(f"Dataset columns: {data.column_names}")
    
    data = data.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': x['answer'],
        'question': x['question']
    })
    return data

def get_minervamath_questions(shot=0, **kwargs) -> Dataset:
    try:
        dataset = load_local_dataset("minervamath", split='test')
    except:
        print("Warning: Local MinervaMath dataset not found, using empty dataset")
        dataset = Dataset.from_dict({'question': [], 'answer': []})
    
    dataset = dataset.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': f"\\boxed{{ {x['answer']} }}",
        'question': x['question'],
    })
    return dataset

def get_amc23_questions(shot=0, **kwargs) -> Dataset:
    try:
        dataset = load_local_dataset("amc23", split='test')
    except:
        print("Warning: Local AMC23 dataset not found, using empty dataset")
        dataset = Dataset.from_dict({'question': [], 'answer': []})
    
    dataset = dataset.map(lambda x: {
        'prompt': get_prompt_shot(x['question'], shot),
        'solution': str(x['answer']),
        'question': x['question'],
    })
    return dataset

def get_arcc_questions(split="test", shot=0, **kwargs) -> Dataset:
    if split == "none":
        split = "test"
    
    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt
    
    def my_char(c):
        mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
        return mapping.get(c, c)
    
    try:
        dataset = load_local_dataset("arc_challenge", split=split)
    except:
        print("Warning: Local ARC Challenge dataset not found, using empty dataset")
        dataset = Dataset.from_dict({'question': [], 'choices': [], 'answerKey': []})
    
    dataset = dataset.map(lambda x: {
        'prompt': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_prompt(x['question'], x['choices']['text']).strip() + '\n' + SYSTEM_PROMPT},
        ],
        'solution': f"\\boxed{{ {my_char(x['answerKey'])} }}",
        'answerKey': my_char(x['answerKey']),
        'question': get_prompt(x['question'], x['choices']['text']).strip(),
    })
    return dataset

#def get_mmlust_questions(shot=0, **kwargs) -> Dataset:
#    def get_prompt(question, choices):
#        prompt = f"Question: {question}\nOptions:\n"
#        for i, choice in enumerate(choices):
#            prompt += f"{chr(65 + i)}. {choice}\n"
#        return prompt
#    
#    dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
#    
#    try:
#        dataset = load_local_dataset("mmlu_stem", split='test')
#    except:
#        print("Warning: Local MMLU-STEM dataset not found, using empty dataset")
#        dataset = Dataset.from_dict({'question': [], 'choices': [], 'answer': []})
#    
#    dataset = dataset.map(lambda x: {
#        'prompt': [
#            {"role": "system", "content": SYSTEM_PROMPT},
#            {"role": "user", "content": get_prompt(x['question'], x['choices']).strip() + '\n' + SYSTEM_PROMPT2},
#        ],
#        'solution': [f"\\boxed{{ {dic[x['answer']]} }}", f"\\boxed{{ {x['choices'][x['answer']]} }}"],
#        'question': get_prompt(x['question'], x['choices']).strip(),
#    })
#    return dataset

def get_mmlust_questions(shot=0, **kwargs) -> Dataset:
    def get_prompt(question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        return prompt
    
    dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    
    try:
        dataset = load_local_dataset("mmlu_stem", split='test')
    except:
        print("Warning: Local MMLU-STEM dataset not found, using empty dataset")
        dataset = Dataset.from_dict({'question': [], 'choices': [], 'answer': []})
    
    dataset = dataset.map(lambda x: {
        'prompt': [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_prompt(x['question'], x['choices']).strip() + '\n' + SYSTEM_PROMPT2},
        ],
        # 修改为字符串格式，只保留选项字母
        'solution': f"\\boxed{{ {dic[x['answer']]} }}",
        'question': get_prompt(x['question'], x['choices']).strip(),
        # 保留原始选项用于后续处理
        'choices': x['choices'],
        'answer_index': x['answer'],
    })
    return dataset

def get_gpqa_diamond_questions(shot=0, **kwargs) -> Dataset:
    """
    加载 GPQA Diamond 数据集
    """
    def get_prompt(question_text):
        """构建 GPQA 格式的 prompt - 明确要求输出选项字母"""
        return [
            {"role": "system", "content": "You are a helpful assistant. Please solve the multiple-choice question. You MUST output ONLY the option letter (A, B, C, or D) as your final answer. Put your final answer in \\boxed{...}."},
            {"role": "user", "content": question_text + "\n\nPlease select the correct option (A, B, C, or D) and put your answer in \\boxed{...}"},
        ]
    
    try:
        dataset = load_local_dataset("gpqa_diamond", split='train', subset='gpqa_diamond.json')
        print(f"Successfully loaded GPQA Diamond dataset with {len(dataset)} examples")
        print(f"GPQA dataset columns: {dataset.column_names}")
        
        if len(dataset) > 0:
            print(f"Sample data: {dataset[0]}")
    except Exception as e:
        print(f"Warning: Local GPQA Diamond dataset not found, using empty dataset. Error: {e}")
        dataset = Dataset.from_dict({'prompt': [], 'final_answer': [], 'choices': []})
    
    if len(dataset) == 0:
        return dataset
    
    # 定义标准选项映射
    OPTION_MAP = {
        'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D',
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D',
    }
    
    def process_example(x):
        prompt_data = x.get('prompt', [])
        if prompt_data and isinstance(prompt_data, list) and len(prompt_data) > 0:
            question_text = prompt_data[0].get('value', '')
        else:
            question_text = x.get('question', x.get('problem', ''))
        
        # 获取答案
        final_answer = x.get('final_answer', x.get('answer', ''))
        
        # 标准化答案 (将 a/b/c/d 转为 A/B/C/D)
        if isinstance(final_answer, str):
            final_answer = final_answer.strip().upper()
            if final_answer in 'ABCD':
                pass  # 已经是标准格式
            elif final_answer in 'abcd':
                final_answer = final_answer.upper()
            else:
                # 尝试从答案中提取选项
                for key in ['A', 'B', 'C', 'D']:
                    if key in final_answer:
                        final_answer = key
                        break
        
        # 获取选项
        choices = x.get('choices', {})
        
        # 构建带选项的问题文本 - 使用标准 A/B/C/D 格式
        if 'Options:' not in question_text and choices:
            options_text = "\n\nOptions:\n"
            # 确保选项按顺序排列
            for letter in ['A', 'B', 'C', 'D']:
                if letter in choices:
                    options_text += f"{letter}. {choices[letter]}\n"
                else:
                    # 如果 choices 使用小写键
                    lower_letter = letter.lower()
                    if lower_letter in choices:
                        options_text += f"{letter}. {choices[lower_letter]}\n"
            full_question = question_text + options_text
        else:
            full_question = question_text
        
        return {
            'prompt': get_prompt(full_question),
            'solution': final_answer,
            'question': question_text,
            'choices': choices,
            'full_question': full_question,
            'answer': final_answer,
        }
    
    dataset = dataset.map(process_example)
    return dataset

# Dataset registry
DATASET_REGISTRY = {
    "gsm8k": get_gsm8k_questions,
    "math500": get_math500_questions,
    "math": get_math_questions,
    "dapo": get_dapo_questions,
    "aime2024": get_aime2024_questions,
    "aime2025": get_aime2025_questions,
    "minervamath": get_minervamath_questions,
    "amc23": get_amc23_questions,
    "arc_challenge": get_arcc_questions,
    "mmlu_stem": get_mmlust_questions,
    "gpqa_diamond": get_gpqa_diamond_questions,  # 新增
}

def get_dataset(name: str, split: str = "train", subset: str = "all", shot: int = 0) -> Dataset:
    """Unified dataset loading interface."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    return DATASET_REGISTRY[name](split=split, subset=subset, shot=shot)