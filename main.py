import os
import nltk


def download_dependencies():
    nltk.download('punkt', download_dir='./nltk_packages')
    nltk.download('stopwords', download_dir='./nltk_packages')
    nltk.download('wordnet', download_dir='./nltk_packages')
    nltk.download('omw-1.4', download_dir='./nltk_packages')


def create_dirs():
    separator = os.path.sep

    current_dir = os.listdir('.')

    algorithms = ['kmeans', 'dbscan', 'gmm']
    dependencies = 'nltk_packages'
    general_subdirectories = ['csvs', 'models', 'figures',
                              f'figures{separator}wordcloud',
                              f'figures{separator}2d',
                              f'figures{separator}3d', ]

    for algo in algorithms:
        if algo not in current_dir or not os.path.isdir(algo):
            os.mkdir(algo)
        for subdir in general_subdirectories:
            full_subdir = f'{algo}{separator}{subdir}'
            if not os.path.isdir(full_subdir):
                os.mkdir(full_subdir)

    if dependencies not in current_dir or not os.path.isdir(dependencies):
        os.mkdir(dependencies)

if __name__ == '__main__':
    create_dirs()
    download_dependencies()