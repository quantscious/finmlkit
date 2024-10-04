 Use the Makefile to build the docs, like so:
   `make builder`
 
## Supported docstring conventions (google)
For documentation generation Sphinx uses the google docstring convention.

[https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html)
```python
def add(a: int, b: int) -> int:
    """
    Adds two numbers.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b
```