from setuptools import setup

package_name = 'vns_system'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Visual Navigation System mock',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vns_node = vns_system.vns_node:main',
        ],
    },
)
