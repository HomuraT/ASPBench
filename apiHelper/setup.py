from setuptools import setup, find_packages

setup(
    name='apiHelper',
    version='0.1.0',
    description='A description of your package',
    long_description=open('readme.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Lin Ren',
    author_email='renlin@seu.edu.cn',
    url='https://github.com/yourusername/your_package',
    packages=find_packages(),  # 指定搜索位置为 'apiHelper'
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'fastapi',
        'pydantic',
        'uvicorn',
        'langchain',
        'langchain-core',
        'langchain-community',
        'langchain-openai>=0.0.10',
        'openai',
        'anthropic',
    ],
)
