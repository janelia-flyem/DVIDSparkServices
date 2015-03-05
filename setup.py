from setuptools import setup

packages=['recospark', 
          'recospark.reconutils',
          'recospark.sparkdvid',
          'workflows'
         ]

package_data={}

setup(name='recospark',
      version='0.1',
      description='Spark-based reconstruction tools working on DVID',
      url='https://github.com/janelia-flyem/reconspark',
      packages=packages,
      package_data=package_data,
      setup_requires=['jsonschema>=1.0', 'pydvid>=0.1', 'argparse', 'importlib']
      )
