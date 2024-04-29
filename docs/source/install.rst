Installation
===================

Install Python
--------------
Same to Cornac, ROSE supports most versions of Python 3. If you have not done so, go to the official `Python download page <https://www.python.org/downloads/>`_.
But we recommend using Python 3.8 or later.

Install ROSE
--------------
We highly recommend using a Python virtual environment to install the packages as some of them (particularly Cython and Sklearn) are only compatible in certain versions. Create a virtual environment within the project's repository (i.e., in /ROSE/). 

0. Optional: Create a virtual environment

.. code-block:: bash

    python3 -m venv rose
    source rose/bin/activate

or 

.. code-block:: bash

    conda create -n rose python=3.11
    conda activate rose

1. Clone the repository from GitHub

.. code-block:: bash

    git clone https://github.com/AthinaKyriakou/ROSE.git
    cd ROSE

2. Install required packages and setup

.. code-block:: bash

    bash setup.sh

3. Build ROSE

.. code-block:: bash

    python setup.py install
    python setup.py build_ext --inplace

Verifying Installation
----------------------
After installing ROSE, you can verify that it has been successfully installed
by running the following command on your favourite terminal/command prompt:

.. code-block:: bash

    python3 -c "import cornac; print(cornac.__version__)"

You should see the following output:

.. parsed-literal::
    |version|

Congratulations! Your machine has ROSE and you're now ready to
create your first experiment!


