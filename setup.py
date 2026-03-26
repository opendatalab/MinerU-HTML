from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(file_path: str) -> list[str]:
    """Read a requirements file and return a list of dependency strings."""
    req_file = Path(__file__).parent / 'requirements' / file_path
    if not req_file.exists():
        return []

    requirements = []
    with open(req_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements


# Core dependencies
core_requirements = read_requirements('core.txt')
openai_requirements = read_requirements('openai.txt')
vllm_requirements = read_requirements('vllm.txt')

# Base install includes only core dependencies
install_requires = core_requirements

# Optional dependency groups
extras_require = {
    'openai': openai_requirements,
    'vllm': vllm_requirements,
    'all': openai_requirements + vllm_requirements,
}

setup(
    name='mineru_html',
    version='1.1.2',
    packages=find_packages(include=['mineru_html*']),
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    description="MinerU-HTML is a main content extraction tool based on Small Language Models.",
    url="https://github.com/opendatalab/MinerU-HTML",
    project_urls={
        'Bug Reports': 'https://github.com/opendatalab/MinerU-HTML/issues',
        'Source': 'https://github.com/opendatalab/MinerU-HTML',
        'Model': 'https://huggingface.co/collections/opendatalab/mineru-html',
    },
    license='Apache License 2.0',
    keywords=['HTML', 'html2text', 'news-crawler' ,'text-extraction', 'scraper', 'webscraping'],
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
)
