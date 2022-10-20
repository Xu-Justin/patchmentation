import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from patchmentation.utils import Comparator

def test_comparator_equal_1():
    object1 = 5
    object2 = 5
    assert Comparator.Equal(object1, object2)

def test_comparator_equal_2():
    object1 = 5
    object2 = 6
    assert not Comparator.Equal(object1, object2)

def test_comparator_greater_1():
    object1 = 4
    object2 = 5
    assert not Comparator.GreaterThan(object1, object2)

def test_comparator_greater_2():
    object1 = 5
    object2 = 5
    assert not Comparator.GreaterThan(object1, object2)

def test_comparator_greater_3():
    object1 = 6
    object2 = 5
    assert Comparator.GreaterThan(object1, object2)

def test_comparator_greater_equal_1():
    object1 = 4
    object2 = 5
    assert not Comparator.GreaterEqual(object1, object2)

def test_comparator_greater_equal_2():
    object1 = 5
    object2 = 5
    assert Comparator.GreaterEqual(object1, object2)

def test_comparator_greater_equal_3():
    object1 = 6
    object2 = 5
    assert Comparator.GreaterEqual(object1, object2)

def test_comparator_less_1():
    object1 = 4
    object2 = 5
    assert Comparator.LessThan(object1, object2)

def test_comparator_less_2():
    object1 = 5
    object2 = 5
    assert not Comparator.LessThan(object1, object2)

def test_comparator_less_3():
    object1 = 6
    object2 = 5
    assert not Comparator.LessThan(object1, object2)

def test_comparator_less_equal_1():
    object1 = 4
    object2 = 5
    assert Comparator.LessEqual(object1, object2)

def test_comparator_less_equal_2():
    object1 = 5
    object2 = 5
    assert Comparator.LessEqual(object1, object2)

def test_comparator_less_equal_3():
    object1 = 6
    object2 = 5
    assert not Comparator.LessEqual(object1, object2)
