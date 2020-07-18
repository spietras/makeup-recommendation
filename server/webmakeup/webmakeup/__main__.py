# now you can import that way:
# from package import module.foo
# or it it's exposed in __init__.py:
# from package import foo

from automakeup.encoded_recommendation import recommend


def main():
    print("webmakeup recommendation for you: ", recommend(1))


if __name__ == '__main__':
    main()
