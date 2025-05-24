from setuptools import setup, find_packages

setup(
    name="meetsage",
    version="0.1",
    description="MeetSage - 智能會議記錄生成與優化工具",
    author="lanss",
    packages=find_packages(),
    package_dir={'': 'src'},
    install_requires=[
        "numpy",
        "rouge_chinese",
        "jieba",
        "bert_score",
        "nltk",
        "pandas",
        "ollama"
    ],
    python_requires='>=3.8',
)
