# put all shortcuts for imports here
# for example if you write this line:
# from .module import foo
# then someone can directly import foo as:
# from package import foo
# instead of:
# from package.module import foo
# this is just a convenience to not expose nesting

from .encoded_recommendation import recommend
package_name = __name__
