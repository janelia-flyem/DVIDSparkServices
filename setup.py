from setuptools import setup

packages=['DVIDSparkServices', 
          'DVIDSparkServices.reconutils',
          'DVIDSparkServices.sparkdvid',
          'workflows'
         ]

package_data={}

setup(name='DVIDSparkServices',
      version='0.1',
      description='Spark-based reconstruction tools working on DVID',
      url='https://github.com/janelia-flyem/DVIDSparkServices',
      packages=packages,
      package_data=package_data,
      setup_requires=['jsonschema>=1.0', 'argparse', 'importlib']
      )
