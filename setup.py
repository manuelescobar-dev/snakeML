from setuptools import setup, find_packages

VERSION = '0.0.6' 
DESCRIPTION = 'Machine Learning Library'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="snakeML", 
        version=VERSION,
        author="Manuel Escobar",
        author_email="manuelescobar.dev@gmail.com",
        description=DESCRIPTION,
        long_description=DESCRIPTION,
        long_description_content_type="text/markdown",
        packages=find_packages(include=['snakeML']),
        install_requires=["numpy","matplotlib","sklearn"], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        license='MIT',
        keywords=['python', 'ML'],
        setup_requires=['pytest-runner'],
        tests_require=['pytest==7.3.1'],
        test_suite='tests',
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)