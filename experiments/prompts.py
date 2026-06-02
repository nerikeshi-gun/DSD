"""
Phase 1D 評価用プロンプト集 (100件)

カテゴリ:
  factual      -- 世界知識の事実補完
  completion   -- 文章・定型表現の補完
  reasoning    -- 簡単な推論・計算
"""

PROMPTS = [
    # --- factual: 地理 (20) ---
    "The capital of Japan is",
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of China is",
    "The capital of Brazil is",
    "The capital of Australia is",
    "The capital of Canada is",
    "The capital of India is",
    "The capital of Russia is",
    "The capital of South Korea is",
    "The largest country in the world is",
    "The largest ocean in the world is",
    "The longest river in the world is",
    "The highest mountain in the world is",
    "The smallest country in the world is",
    "The largest continent is",
    "The deepest lake in the world is",
    "The largest desert in the world is",

    # --- factual: 科学・自然 (20) ---
    "The speed of light is approximately",
    "Water freezes at",
    "Water boils at",
    "The chemical symbol for gold is",
    "The chemical symbol for iron is",
    "The chemical symbol for water is",
    "The first element in the periodic table is",
    "The largest planet in the solar system is",
    "The closest planet to the sun is",
    "The number of planets in the solar system is",
    "The atomic number of carbon is",
    "DNA stands for",
    "The speed of sound is approximately",
    "The human body has",
    "Photosynthesis converts sunlight into",
    "The powerhouse of the cell is the",
    "Gravity on the surface of Earth is approximately",
    "The sun is a",
    "Light travels at",
    "The boiling point of nitrogen is",

    # --- factual: テクノロジー・CS (20) ---
    "Python is a",
    "Linux is",
    "The inventor of the telephone was",
    "HTML stands for",
    "CPU stands for",
    "The first programming language was",
    "Java is a",
    "Git is a",
    "An algorithm is",
    "Machine learning is a subset of",
    "The internet was invented in",
    "Binary code uses",
    "A database is",
    "Open source software means",
    "The Linux kernel was created by",
    "Moore's law states that",
    "TCP/IP stands for",
    "A compiler converts",
    "An API is",
    "RAM stands for",

    # --- factual: 歴史・文化 (15) ---
    "World War II ended in",
    "The first moon landing was in",
    "The Berlin Wall fell in",
    "Shakespeare was born in",
    "The French Revolution began in",
    "The Great Wall of China was built to",
    "Columbus reached America in",
    "The Renaissance began in",
    "Einstein developed the theory of",
    "The Olympic Games originated in",
    "The printing press was invented by",
    "The United Nations was founded in",
    "Penicillin was discovered by",
    "The first Nobel Prize was awarded in",
    "The Roman Empire fell in",

    # --- completion: 定型・慣用表現 (15) ---
    "The opposite of hot is",
    "The opposite of cold is",
    "The opposite of fast is",
    "The opposite of dark is",
    "The opposite of open is",
    "The sun rises in the",
    "The sun sets in the",
    "A stitch in time saves",
    "Actions speak louder than",
    "Every cloud has a silver",
    "The early bird catches the",
    "Two plus two equals",
    "The square root of 144 is",
    "A dozen is equal to",
    "There are 24 hours in a",

    # --- reasoning: 簡単な推論 (10) ---
    "If all humans are mortal and Socrates is human, then Socrates is",
    "The next number in the sequence 2, 4, 6, 8 is",
    "If today is Monday, tomorrow is",
    "A triangle has",
    "A square has",
    "The sum of angles in a triangle is",
    "Water is composed of hydrogen and",
    "If a car travels 60 miles per hour for 2 hours, it covers",
    "To convert Celsius to Fahrenheit, you multiply by 9/5 and add",
    "The plural of mouse is",
]

assert len(PROMPTS) >= 100, f"Expected 100+ prompts, got {len(PROMPTS)}"
