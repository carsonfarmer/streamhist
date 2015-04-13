Contributions to StreamHist are very welcome. They are likely to be
accepted more quickly if they follow these guidelines.

Conventions
~~~~~~~~~~~

In general, StreamHist follows the style and documentation conventions
of the Numpy project where applicable. Please see `A Guide to
NumPy/SciPy
Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`__
for details.

Documentation
~~~~~~~~~~~~~

Classes, methods, functions, etc. should have docstrings. The first line
of a docstring should be a standalone summary. Parameters and return
values should be documented explicitly. Use the following sections where
appropriate for docstrings:

1.  Summary
2.  Extended Summary (optional)
3.  Parameters
4.  Returns
5.  Other parameters (optional)
6.  Raises (optional)
7.  See Also (optional)
8.  Notes (optional)
9.  References (optional)
10. Examples (optional, but strongly encouraged!)

**Style**

-  For readibility, use *italics*, **bold** and ``monospace`` if needed
   in any explanations (but not for variable names and doctest code or
   multi-line code). Variable, module, function, and class names should
   be written between single back-ticks (``protoboard``).

-  The optional ``Examples`` section should use the
   `doctest <http://docs.python.org/library/doctest.html>`__ format.
   This section is meant to illustrate usage, not to provide a testing
   framework -- for that, use the tests/ directory. While optional, this
   section is very strongly encouraged.

-  Each module should have a docstring with at least a summary line.
   Other sections are optional, and should be used in the same order as
   for documenting functions etc. when they are appropriate. You may
   also want to include a ``Routine Listings`` section.

Versions
~~~~~~~~

StreamHist will use `Semantic Versioning <http://semver.org>`__ as our
formal convention for specifying compatibility, using a three-part
version number: major version; minor version; and patch. Additional
labels for pre-release and build metadata may be appended to the
three-part format. The currently installed version of StreamHist can be
checked by printing ``streamhist.__version__``.

Support
~~~~~~~

StreamHist supports python 2 (2.6+) and python 3 (3.2+) with a single
code base. Use modern python idioms when possible that are compatibile
with both major versions, and use the `six <https://pythonhosted.org/six/>`__
library where helpful to smooth over the differences. Use
``from __future__ import`` statements where appropriate. Test code locally in
both python 2 and python 3 when possible.

Testing
~~~~~~~

Currently, StreamHist has minimal test coverage; this will be
changed soon! Where possible, adopt a `Test Driven Development
(TDD) <http://en.wikipedia.org/wiki/Test-driven_development>`__ software
development process.

In general, new functionality should include tests. Please write
reasonable tests for your code and make sure that they pass, while at
the same time, ensuring that all existing tests also pass.
