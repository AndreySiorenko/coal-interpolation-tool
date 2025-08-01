"""
Mock implementations for external dependencies.
Used when libraries like pandas, numpy, scipy, matplotlib are not available.
"""
import math
import random
from typing import List, Tuple, Any, Optional, Dict


class MockDataFrame:
    """Mock implementation of pandas.DataFrame"""
    
    def __init__(self, data=None, columns=None):
        if data is None:
            self.data = []
            self.columns = columns or []
        elif isinstance(data, dict):
            self.columns = list(data.keys())
            self.data = [list(row) for row in zip(*data.values())]
        else:
            self.data = data
            self.columns = columns or [f"col_{i}" for i in range(len(data[0]) if data else 0)]
    
    def __getitem__(self, key):
        if key in self.columns:
            col_idx = self.columns.index(key)
            return [row[col_idx] for row in self.data]
        return None
    
    def __len__(self):
        return len(self.data)
    
    def head(self, n=5):
        return MockDataFrame(self.data[:n], self.columns)
    
    def describe(self):
        # Simple mock statistics
        return {"count": len(self.data), "mean": 0.0, "min": 0.0, "max": 1.0}
    
    def to_dict(self, orient='dict'):
        if orient == 'records':
            return [{col: row[i] for i, col in enumerate(self.columns)} for row in self.data]
        return {col: [row[i] for row in self.data] for i, col in enumerate(self.columns)}
    
    @property
    def shape(self):
        return (len(self.data), len(self.columns))


class MockArray:
    """Mock implementation of numpy.array"""
    
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            self.data = list(data)
        else:
            self.data = [data]
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    @property
    def shape(self):
        return (len(self.data),)
    
    def min(self):
        return min(self.data) if self.data else 0
    
    def max(self):
        return max(self.data) if self.data else 0
    
    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0


class MockNumpy:
    """Mock implementation of numpy module"""
    
    @staticmethod
    def array(data):
        return MockArray(data)
    
    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            return MockArray([0.0] * shape)
        return MockArray([0.0] * (shape[0] * shape[1]))
    
    @staticmethod
    def ones(shape):
        if isinstance(shape, int):
            return MockArray([1.0] * shape)
        return MockArray([1.0] * (shape[0] * shape[1]))
    
    @staticmethod
    def linspace(start, stop, num):
        if num <= 1:
            return MockArray([start])
        step = (stop - start) / (num - 1)
        return MockArray([start + i * step for i in range(num)])
    
    @staticmethod
    def meshgrid(x, y):
        # Простая реализация meshgrid
        x_data = x.data if hasattr(x, 'data') else x
        y_data = y.data if hasattr(y, 'data') else y
        X = MockArray([x_data for _ in y_data])
        Y = MockArray([y_data for _ in x_data])
        return X, Y
    
    @staticmethod
    def sqrt(x):
        if hasattr(x, 'data'):
            return MockArray([math.sqrt(val) for val in x.data])
        return math.sqrt(x)
    
    @staticmethod
    def isnan(x):
        if hasattr(x, 'data'):
            return MockArray([math.isnan(val) if isinstance(val, float) else False for val in x.data])
        return math.isnan(x) if isinstance(x, float) else False


class MockPandas:
    """Mock implementation of pandas module"""
    
    DataFrame = MockDataFrame
    
    @staticmethod
    def read_csv(filepath, **kwargs):
        # Создаем фиктивный DataFrame с тестовыми данными
        return MockDataFrame({
            'X': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Z': [10.0, 15.0, 20.0, 25.0, 30.0]
        })


class MockKDTree:
    """Mock implementation of scipy.spatial.cKDTree"""
    
    def __init__(self, data):
        self.data = data
    
    def query(self, point, k=1, distance_upper_bound=float('inf')):
        # Простая реализация поиска ближайших соседей
        distances = []
        indices = []
        
        for i, data_point in enumerate(self.data):
            if hasattr(data_point, 'data'):
                data_point = data_point.data
            
            # Вычисляем евклидово расстояние
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, data_point)))
            
            if dist <= distance_upper_bound:
                distances.append(dist)
                indices.append(i)
        
        # Сортируем по расстоянию и берем k ближайших
        sorted_pairs = sorted(zip(distances, indices))[:k]
        
        if k == 1:
            return sorted_pairs[0] if sorted_pairs else (float('inf'), len(self.data))
        
        distances, indices = zip(*sorted_pairs) if sorted_pairs else ([], [])
        return list(distances), list(indices)


class MockScipy:
    """Mock implementation of scipy module"""
    
    class spatial:
        cKDTree = MockKDTree


class MockMatplotlib:
    """Mock implementation of matplotlib module"""
    
    class pyplot:
        @staticmethod
        def figure(*args, **kwargs):
            return MockFigure()
        
        @staticmethod
        def subplots(*args, **kwargs):
            return MockFigure(), MockAxes()
        
        @staticmethod
        def show():
            print("Mock: matplotlib.pyplot.show() called")
        
        @staticmethod
        def savefig(filename, **kwargs):
            print(f"Mock: Saving figure to {filename}")
        
        @staticmethod
        def close(*args):
            print("Mock: matplotlib.pyplot.close() called")


class MockFigure:
    """Mock implementation of matplotlib.Figure"""
    
    def __init__(self):
        self.axes = [MockAxes()]
    
    def add_subplot(self, *args, **kwargs):
        return MockAxes()
    
    def savefig(self, filename, **kwargs):
        print(f"Mock: Saving figure to {filename}")
    
    def clear(self):
        print("Mock: Figure cleared")


class MockAxes:
    """Mock implementation of matplotlib.Axes"""
    
    def scatter(self, x, y, **kwargs):
        print(f"Mock: Scatter plot with {len(x) if hasattr(x, '__len__') else 1} points")
        return self
    
    def contour(self, X, Y, Z, **kwargs):
        print("Mock: Contour plot created")
        return self
    
    def contourf(self, X, Y, Z, **kwargs):
        print("Mock: Filled contour plot created")
        return self
    
    def set_xlabel(self, label):
        print(f"Mock: X label set to '{label}'")
    
    def set_ylabel(self, label):
        print(f"Mock: Y label set to '{label}'")
    
    def set_title(self, title):
        print(f"Mock: Title set to '{title}'")
    
    def grid(self, *args, **kwargs):
        print("Mock: Grid enabled")
    
    def legend(self, *args, **kwargs):
        print("Mock: Legend added")
    
    def colorbar(self, *args, **kwargs):
        print("Mock: Colorbar added")
        return self
    
    def clear(self):
        print("Mock: Axes cleared")


def setup_mock_environment():
    """
    Настройка mock-окружения для работы без внешних зависимостей.
    Возвращает словарь с mock-модулями.
    """
    return {
        'pandas': MockPandas(),
        'numpy': MockNumpy(),
        'scipy': MockScipy(),
        'matplotlib': MockMatplotlib()
    }


def get_mock_sample_data():
    """Создает тестовые данные для демонстрации"""
    return MockDataFrame({
        'X': [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        'Y': [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5],
        'Z': [10.0, 15.0, 20.0, 25.0, 30.0, 12.5, 17.5, 22.5, 27.5]
    })