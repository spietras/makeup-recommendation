import pathlib
import shutil
import tempfile
import unittest

from extract_wheels.lib import namespace_pkgs


class TempDir:
    def __init__(self):
        self.dir = tempfile.mkdtemp()

    def root(self):
        return self.dir

    def add_dir(self, rel_path):
        d = pathlib.Path(self.dir, rel_path)
        d.mkdir(parents=True)

    def add_file(self, rel_path, contents=None):
        f = pathlib.Path(self.dir, rel_path)
        f.parent.mkdir(parents=True, exist_ok=True)
        if contents:
            with open(f, "w") as writeable_f:
                writeable_f.write(contents)
        else:
            f.touch()

    def remove(self):
        shutil.rmtree(self.dir)


class TestPkgResourcesStyleNamespacePackages(unittest.TestCase):
    def test_finds_correct_namespace_packages(self):
        directory = TempDir()
        directory.add_file("google/auth/__init__.py")
        directory.add_file("google/auth/foo.py")
        directory.add_file(
            "google_auth-1.8.2.dist-info/namespace_packages.txt", contents="google\n"
        )

        expected = {
            directory.root() + "/google",
        }
        actual = namespace_pkgs.pkg_resources_style_namespace_packages(directory.root())
        self.assertEqual(actual, expected)

    def test_nested_namespace_packages(self):
        directory = TempDir()
        directory.add_file("google/auth/__init__.py")
        directory.add_file("google/auth/foo.py")
        directory.add_file("google/bar/biz/__init__.py")
        directory.add_file("google/bar/biz/bee.py")
        directory.add_file(
            "google_auth-1.8.2.dist-info/namespace_packages.txt",
            contents="google\ngoogle.bar\n",
        )

        expected = {
            directory.root() + "/google",
            directory.root() + "/google/bar",
        }
        actual = namespace_pkgs.pkg_resources_style_namespace_packages(directory.root())
        self.assertEqual(actual, expected)

    def test_empty_case(self):
        # Even though this directory contains directories with no __init__.py
        # it has an empty namespace_packages.txt file so no namespace packages
        # should be returned.
        directory = TempDir()
        directory.add_file("foo/bar/biz.py")
        directory.add_file("foo/bee/boo.py")
        directory.add_file("foo/buu/__init__.py")
        directory.add_file("foo/buu/bii.py")
        directory.add_file("foo-1.0.0.dist-info/namespace_packages.txt")

        actual = namespace_pkgs.pkg_resources_style_namespace_packages(directory.root())
        self.assertEqual(actual, set())

    def test_missing_namespace_pkgs_record_file(self):
        # Even though this directory contains directories with no __init__.py
        # it has no namespace_packages.txt file, so no namespace packages should
        # be found and returned.
        directory = TempDir()
        directory.add_file("foo/bar/biz.py")
        directory.add_file("foo/bee/boo.py")
        directory.add_file("foo/buu/__init__.py")
        directory.add_file("foo/buu/bii.py")
        directory.add_file("foo-1.0.0.dist-info/METADATA")
        directory.add_file("foo-1.0.0.dist-info/RECORD")

        actual = namespace_pkgs.pkg_resources_style_namespace_packages(directory.root())
        self.assertEqual(actual, set())


class TestImplicitNamespacePackages(unittest.TestCase):
    def test_finds_correct_namespace_packages(self):
        directory = TempDir()
        directory.add_file("foo/bar/biz.py")
        directory.add_file("foo/bee/boo.py")
        directory.add_file("foo/buu/__init__.py")
        directory.add_file("foo/buu/bii.py")

        expected = {
            directory.root() + "/foo",
            directory.root() + "/foo/bar",
            directory.root() + "/foo/bee",
        }
        actual = namespace_pkgs.implicit_namespace_packages(directory.root())
        self.assertEqual(actual, expected)

    def test_ignores_empty_directories(self):
        directory = TempDir()
        directory.add_file("foo/bar/biz.py")
        directory.add_dir("foo/cat")

        expected = {
            directory.root() + "/foo",
            directory.root() + "/foo/bar",
        }
        actual = namespace_pkgs.implicit_namespace_packages(directory.root())
        self.assertEqual(actual, expected)

    def test_empty_case(self):
        directory = TempDir()
        directory.add_file("foo/__init__.py")
        directory.add_file("foo/bar/__init__.py")
        directory.add_file("foo/bar/biz.py")

        actual = namespace_pkgs.implicit_namespace_packages(directory.root())
        self.assertEqual(actual, set())


class TestaddPkgutilStyleNamespacePkgInit(unittest.TestCase):
    def test_missing_directory_is_created(self):
        missing_directory = pathlib.Path(TempDir().root(), "missing_directory")
        namespace_pkgs.add_pkgutil_style_namespace_pkg_init(str(missing_directory))
        self.assertTrue(missing_directory.is_dir())


if __name__ == "__main__":
    unittest.main()
