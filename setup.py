from setuptools import find_packages, setup

setup( name='DVIDSparkServices',
       version='0.3',
       description='Spark-based reconstruction tools working on DVID',
       url='https://github.com/janelia-flyem/DVIDSparkServices',
       packages=find_packages(exclude=('unit_tests', 'integration_tests')),
       package_data={'DVIDSparkServices': ['SPARK_CONF_DIR/*']},
       test_suite="DVIDSparkServices.reconutils.metrics.tests",
       entry_points={
          'console_scripts': [
              'launchworkflow = DVIDSparkServices.workflow.launchworkflow:main',
              'sparklaunch_janelia_lsf = DVIDSparkServices.spark_launch_scripts.janelia_lsf.sparklaunch_janelia_lsf:main',
              'sparklaunch_janelia_lsf_int = DVIDSparkServices.spark_launch_scripts.janelia_lsf.sparklaunch_janelia_lsf_int:main'
          ]
       }
     )
