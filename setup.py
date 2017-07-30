from setuptools import setup

packages=['DVIDSparkServices', 
          'DVIDSparkServices.reconutils',
          'DVIDSparkServices.reconutils.plugins',
          'DVIDSparkServices.io_util',
          'DVIDSparkServices.dvid',
          'DVIDSparkServices.sparkdvid',
          'DVIDSparkServices.workflow',
          'DVIDSparkServices.workflows'
         ]

package_data={}

setup( name='DVIDSparkServices',
       version='0.3',
       description='Spark-based reconstruction tools working on DVID',
       url='https://github.com/janelia-flyem/DVIDSparkServices',
       packages=packages,
       package_data=package_data,
       entry_points={
          'console_scripts': [
              'launchworkflow = DVIDSparkServices.workflow.launchworkflow:main'
          ]
       }
     )
