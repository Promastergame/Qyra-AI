"""
generate_finetune_data.py — Генерация расширенного finetune dataset.

Создаёт 3000+ диалогов с:
  - Обычными вопросами/ответами
  - Tool calls (calculator, python)
  - Разнообразными темами

Usage:
    python generate_finetune_data.py [--output data/finetune/chat.jsonl]
"""

import json
import random
import os
from config import DATA_FT_DIR

random.seed(42)

# --- Шаблоны диалогов ---

# Простые вопросы/ответы (без tools)
SIMPLE_QA = [
    ("What is Python?", "Python is a high-level programming language known for its clear syntax and versatility."),
    ("What is machine learning?", "Machine learning is a branch of AI where computers learn from data without explicit programming."),
    ("What is a neural network?", "A neural network is a computing system inspired by biological neurons, consisting of layers of interconnected nodes."),
    ("What is an API?", "An API is a set of rules that allows different software applications to communicate with each other."),
    ("What is recursion?", "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem."),
    ("What is Linux?", "Linux is an open-source operating system kernel used in servers, desktops, and embedded devices."),
    ("What is Git?", "Git is a distributed version control system for tracking changes in source code."),
    ("What is Docker?", "Docker is a platform for developing, shipping, and running applications in containers."),
    ("What is Kubernetes?", "Kubernetes is an open-source container orchestration platform."),
    ("What is REST API?", "REST is an architectural style for designing networked applications using HTTP requests."),
    ("What is SQL?", "SQL is a language for managing and querying relational databases."),
    ("What is NoSQL?", "NoSQL databases store data in flexible formats like documents or key-value pairs."),
    ("What is HTTPS?", "HTTPS is a secure version of HTTP that encrypts data between client and server."),
    ("What is DNS?", "DNS translates domain names into IP addresses."),
    ("What is a firewall?", "A firewall monitors and controls network traffic based on security rules."),
    ("What is encryption?", "Encryption converts data into a code to prevent unauthorized access."),
    ("What is authentication?", "Authentication verifies the identity of a user or system."),
    ("What is authorization?", "Authorization determines what resources a user can access."),
    ("What is a variable?", "A variable is a named storage location in programming that holds a value."),
    ("What is a function?", "A function is a reusable block of code that performs a specific task."),
    ("What is a loop?", "A loop repeatedly executes a block of code until a condition is met."),
    ("What is an array?", "An array is a data structure that stores a collection of elements."),
    ("What is a class?", "A class is a blueprint for creating objects in object-oriented programming."),
    ("What is inheritance?", "Inheritance allows a class to inherit properties and methods from another class."),
    ("What is polymorphism?", "Polymorphism allows objects to be treated as instances of their parent class."),
    ("What is encapsulation?", "Encapsulation bundles data and methods that operate on that data within a class."),
    ("What is abstraction?", "Abstraction hides complex implementation details and shows only essential features."),
    ("What is debugging?", "Debugging is the process of finding and fixing errors in code."),
    ("What is testing?", "Testing involves running a program to find bugs and verify it works correctly."),
    ("What is CI/CD?", "CI/CD automates the integration and deployment of code changes."),
    ("What is microservices?", "Microservices architecture builds applications as small, independent services."),
    ("What is cloud computing?", "Cloud computing delivers computing services over the internet."),
    ("What is edge computing?", "Edge computing processes data closer to where it's generated."),
    ("What is IoT?", "IoT connects everyday devices to the internet for data exchange."),
    ("What is blockchain?", "Blockchain is a distributed ledger technology for secure transactions."),
    ("What is cryptocurrency?", "Cryptocurrency is a digital currency secured by cryptography."),
    ("What is big data?", "Big data refers to extremely large datasets analyzed for patterns and insights."),
    ("What is data science?", "Data science combines statistics and programming to extract insights from data."),
    ("What is deep learning?", "Deep learning uses multi-layered neural networks to learn complex patterns."),
    ("What is NLP?", "NLP enables computers to understand and generate human language."),
    ("What is computer vision?", "Computer vision enables machines to interpret visual information."),
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there!", "Hi! What can I do for you?"),
    ("Good morning!", "Good morning! How can I assist you?"),
    ("Good evening!", "Good evening! What would you like to know?"),
    ("Thank you!", "You're welcome! Let me know if you need anything else."),
    ("Thanks!", "Happy to help! Feel free to ask more questions."),
    ("What's up?", "Not much! How about you? What would you like to learn?"),
    ("How are you?", "I'm doing great! Ready to answer your questions."),
]

# Математические вопросы (с calculator tool)
MATH_QUESTIONS = [
    ("Calculate 2 + 2", "4"),
    ("Calculate 5 * 7", "35"),
    ("Calculate 10 + 15", "25"),
    ("Calculate 20 * 30", "600"),
    ("Calculate 100 - 37", "63"),
    ("Calculate 144 / 12", "12"),
    ("Calculate 15 + 25", "40"),
    ("Calculate 8 * 9", "72"),
    ("Calculate 56 / 8", "7"),
    ("Calculate 1000 - 250", "750"),
    ("Calculate 12 * 12", "144"),
    ("Calculate 7 + 8 + 9", "24"),
    ("Calculate 50 * 2", "100"),
    ("Calculate 36 / 6", "6"),
    ("Calculate 100 + 200", "300"),
    ("Calculate 9 * 9", "81"),
    ("Calculate 64 / 8", "8"),
    ("Calculate 75 - 25", "50"),
    ("Calculate 11 * 11", "121"),
    ("Calculate 81 / 9", "9"),
    ("Calculate 25 + 75", "100"),
    ("Calculate 6 * 7", "42"),
    ("Calculate 48 / 6", "8"),
    ("Calculate 90 - 45", "45"),
    ("Calculate 13 * 5", "65"),
    ("Calculate 100 / 4", "25"),
    ("Calculate 18 + 22", "40"),
    ("Calculate 4 * 25", "100"),
    ("Calculate 63 / 7", "9"),
    ("Calculate 200 - 100", "100"),
    ("Calculate 14 * 3", "42"),
    ("Calculate 72 / 8", "9"),
    ("Calculate 33 + 67", "100"),
    ("Calculate 5 * 5 * 5", "125"),
    ("Calculate 1000 / 10", "100"),
    ("Calculate 17 + 28 + 35", "80"),
    ("Calculate 9 * 8", "72"),
    ("Calculate 56 - 28", "28"),
    ("Calculate 16 * 4", "64"),
    ("Calculate 96 / 12", "8"),
    ("Calculate 45 + 55", "100"),
    ("Calculate 7 * 8", "56"),
    ("Calculate 84 / 7", "12"),
    ("Calculate 150 - 75", "75"),
    ("Calculate 23 * 4", "92"),
    ("Calculate 65 / 5", "13"),
    ("Calculate 38 + 62", "100"),
    ("Calculate 12 * 10", "120"),
    ("Calculate 77 / 7", "11"),
    ("Calculate 250 - 125", "125"),
]

# Python задачи (с python tool)
PYTHON_TASKS = [
    ("Write python hello world", "print(\"hello world\")"),
    ("Write python to add two numbers", "a = 5\nb = 10\nprint(a + b)"),
    ("Write python to print numbers 1 to 5", "for i in range(1, 6):\n    print(i)"),
    ("Write python to check if number is even", "n = 10\nif n % 2 == 0:\n    print(\"even\")\nelse:\n    print(\"odd\")"),
    ("Write python to reverse a string", "s = \"hello\"\nprint(s[::-1])"),
    ("Write python to find max in list", "nums = [3, 7, 2, 9, 1]\nprint(max(nums))"),
    ("Write python to calculate factorial", "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)\nprint(factorial(5))"),
    ("Write python to print fibonacci", "a, b = 0, 1\nfor _ in range(6):\n    print(a)\n    a, b = b, a + b"),
    ("Write python to check palindrome", "s = \"radar\"\nprint(s == s[::-1])"),
    ("Write python to count vowels", "s = \"hello\"\nvowels = \"aeiou\"\nprint(sum(1 for c in s if c in vowels))"),
    ("Write python to swap two variables", "a, b = 5, 10\na, b = b, a\nprint(a, b)"),
    ("Write python to find prime numbers", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\nprint([x for x in range(2, 20) if is_prime(x)])"),
    ("Write python to sort a list", "nums = [5, 2, 8, 1, 9]\nprint(sorted(nums))"),
    ("Write python to find average", "nums = [10, 20, 30, 40]\nprint(sum(nums) / len(nums))"),
    ("Write python to remove duplicates", "lst = [1, 2, 2, 3, 3, 3]\nprint(list(set(lst)))"),
    ("Write python to multiply list elements", "from functools import reduce\nnums = [1, 2, 3, 4]\nprint(reduce(lambda x, y: x * y, nums))"),
    ("Write python to check leap year", "year = 2024\nif (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):\n    print(\"leap\")\nelse:\n    print(\"not leap\")"),
    ("Write python to generate random number", "import random\nprint(random.randint(1, 100)"),
    ("Write python to convert celsius to fahrenheit", "c = 25\nf = (c * 9/5) + 32\nprint(f)"),
    ("Write python to find gcd", "import math\nprint(math.gcd(48, 18))"),
    ("Write python to print multiplication table", "n = 5\nfor i in range(1, 11):\n    print(f\"{n} x {i} = {n*i}\")"),
    ("Write python to count words", "s = \"hello world hello\"\nprint(len(s.split()))"),
    ("Write python to find length of string", "s = \"python\"\nprint(len(s))"),
    ("Write python to check armstrong number", "n = 153\nprint(n == sum(int(d)**3 for d in str(n)))"),
    ("Write python to merge two dictionaries", "d1 = {'a': 1}\nd2 = {'b': 2}\nprint({**d1, **d2})"),
]

# Объяснения программирования
CODE_EXPLANATIONS = [
    ("explain loop", "A loop repeatedly executes a block of code. In Python, 'for' loops iterate over sequences, while 'while' loops continue until a condition becomes false."),
    ("explain function", "A function is a reusable code block. It takes inputs (parameters), performs operations, and optionally returns a result."),
    ("explain variable", "A variable stores data that can be used and modified in a program. Variables have names and types."),
    ("explain array", "An array stores multiple values in a single variable. Elements are accessed by their index position."),
    ("explain recursion", "Recursion is when a function calls itself. It needs a base case to stop, otherwise it would run infinitely."),
    ("explain OOP", "Object-Oriented Programming organizes code into objects that contain data and methods. Key concepts: encapsulation, inheritance, polymorphism."),
    ("explain class", "A class is a blueprint for creating objects. It defines attributes (data) and methods (functions) that objects will have."),
    ("explain inheritance", "Inheritance allows a class to inherit attributes and methods from a parent class, promoting code reuse."),
    ("explain polymorphism", "Polymorphism allows different classes to be treated as instances of the same base class, with method overriding."),
    ("explain encapsulation", "Encapsulation bundles data and methods within a class, hiding internal details from outside access."),
    ("explain API", "An API defines how software components interact. It specifies requests, responses, and data formats."),
    ("explain database", "A database stores structured data. SQL databases use tables; NoSQL databases use documents, graphs, or key-value pairs."),
    ("explain HTTP", "HTTP is the protocol for web communication. It uses methods like GET, POST, PUT, DELETE for CRUD operations."),
    ("explain Git", "Git is a version control system. Key commands: commit (save), push (upload), pull (download), branch (parallel development)."),
    ("explain Docker", "Docker packages applications in containers with all dependencies, ensuring consistent environments."),
    ("explain CI/CD", "CI/CD automates code integration, testing, and deployment, enabling rapid and reliable releases."),
    ("explain microservices", "Microservices break applications into small, independent services that communicate via APIs."),
    ("explain cloud", "Cloud computing provides on-demand computing resources over the internet, like servers, storage, and databases."),
    ("explain machine learning", "Machine learning trains models on data to make predictions. Types: supervised, unsupervised, reinforcement learning."),
    ("explain neural network", "A neural network has layers of neurons. Each neuron processes inputs and passes results through activation functions."),
    ("explain deep learning", "Deep learning uses neural networks with many layers to learn hierarchical representations from data."),
    ("explain NLP", "Natural Language Processing enables computers to understand, interpret, and generate human language."),
    ("explain computer vision", "Computer vision enables machines to extract meaning from images and videos."),
    ("explain blockchain", "Blockchain is a distributed ledger where transactions are recorded in blocks linked by cryptography."),
    ("explain encryption", "Encryption transforms data into unreadable format. Only authorized parties with the key can decrypt it."),
]


def create_simple_dialogue(question, answer):
    """Создать простой диалог без tools."""
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }


def create_calculator_dialogue(question, result):
    """Создать диалог с calculator tool."""
    # Извлекаем выражение из вопроса
    expr = question.replace("Calculate ", "").strip()
    
    intros = [
        "Let me calculate that for you.",
        "I'll compute that.",
        "Let me do the math.",
        "Calculating...",
        "I'll calculate that.",
    ]
    
    outros = [
        f"The result is {result}.",
        f"That equals {result}.",
        f"The answer is {result}.",
        f"It's {result}.",
        f"Result: {result}.",
    ]
    
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with access to a calculator tool."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": random.choice(intros)},
            {"role": "tool_call", "content": f"calculator\n{{\"expression\": \"{expr}\"}}"},
            {"role": "tool_result", "content": str(result)},
            {"role": "assistant", "content": random.choice(outros)}
        ]
    }


def create_python_dialogue(task, code):
    """Создать диалог с python tool."""
    intros = [
        "Sure, I can write that code for you.",
        "I'll write and run that code.",
        "Let me write the Python code.",
        "Here's the code:",
        "I'll create that for you.",
    ]
    
    outros = [
        "The code executed successfully.",
        "Here's the result.",
        "The script ran successfully.",
        "Code completed.",
        "Done! Here's what it outputs.",
    ]
    
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with a Python tool."},
            {"role": "user", "content": task},
            {"role": "assistant", "content": random.choice(intros)},
            {"role": "tool_call", "content": f"python\n{{\"code\": {json.dumps(code)}}}"},
            {"role": "tool_result", "content": "Code executed."},
            {"role": "assistant", "content": random.choice(outros)}
        ]
    }


def create_explanation_dialogue(question, answer):
    """Создать диалог с объяснением."""
    intros = [
        "Sure, let me explain.",
        "Here's an explanation.",
        "Let me clarify that.",
        "Good question! Here's the explanation.",
    ]
    
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": random.choice(intros) + " " + answer}
        ]
    }


def generate_dataset(num_simple=1500, num_math=1000, num_python=500, num_explanations=500):
    """Сгенерировать полный датасет."""
    dialogues = []
    
    # Простые диалоги (повторяем шаблоны с вариациями)
    print(f"Generating {num_simple} simple dialogues...")
    for i in range(num_simple):
        q, a = random.choice(SIMPLE_QA)
        # Добавляем небольшие вариации
        if random.random() < 0.3:
            q = q.rstrip("?") + "?"
        if random.random() < 0.2:
            a = "Sure! " + a
        dialogues.append(create_simple_dialogue(q, a))
    
    # Математические диалоги
    print(f"Generating {num_math} calculator dialogues...")
    for i in range(num_math):
        q, result = random.choice(MATH_QUESTIONS)
        dialogues.append(create_calculator_dialogue(q, result))
    
    # Python диалоги
    print(f"Generating {num_python} Python dialogues...")
    for i in range(num_python):
        task, code = random.choice(PYTHON_TASKS)
        dialogues.append(create_python_dialogue(task, code))
    
    # Объяснения
    print(f"Generating {num_explanations} explanation dialogues...")
    for i in range(num_explanations):
        q, a = random.choice(CODE_EXPLANATIONS)
        dialogues.append(create_explanation_dialogue(q, a))
    
    # Перемешиваем
    random.shuffle(dialogues)
    
    return dialogues


def main():
    output_path = os.path.join(DATA_FT_DIR, "chat.jsonl")
    
    print("=" * 50)
    print("Generating fine-tune dataset")
    print("=" * 50)
    
    # Генерируем 3500 диалогов
    dialogues = generate_dataset(
        num_simple=1500,
        num_math=1000,
        num_python=500,
        num_explanations=500
    )
    
    print(f"\nTotal dialogues: {len(dialogues)}")
    
    # Сохраняем
    os.makedirs(DATA_FT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for dlg in dialogues:
            f.write(json.dumps(dlg, ensure_ascii=False) + "\n")
    
    print(f"Saved to {output_path}")
    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
