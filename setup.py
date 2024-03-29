import setuptools

import site
import sys


if __name__ == "__main__":
    site.ENABLE_USER_SITE = 1
    setuptools.setup()