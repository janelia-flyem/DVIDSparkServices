from setuptools import setup

packages=['DVIDSparkServices', 
          'DVIDSparkServices.reconutils',
          'DVIDSparkServices.reconutils.plugins',
          'DVIDSparkServices.sparkdvid',
          'DVIDSparkServices.workflow',
          'workflows'
         ]

package_data={}

setup(name='DVIDSparkServices',
      version='0.1',
      description='Spark-based reconstruction tools working on DVID',
      url='https://github.com/janelia-flyem/DVIDSparkServices',
      packages=packages,
      package_data=package_data
      )
