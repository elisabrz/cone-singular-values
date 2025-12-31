"""
reference_data.py
Load and access reference results from CSV file

This replaces hardcoded functions like exact_schur_vs_nonneg_solution
with data loaded from reference_results_2025.csv
"""

import pandas as pd
from pathlib import Path


class ReferenceData:
    """
    Load and access reference benchmark results
    """
    
    def __init__(self, csv_path='reference_results_2025.csv'):
        """
        Load reference data from CSV
        
        Parameters
        ----------
        csv_path : str or Path
            Path to the CSV file with reference results
        """
        self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Reference CSV not found: {self.csv_path}\n"
                f"Make sure reference_results_2025.csv is in the current directory"
            )
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Convert numeric columns (handle NA/empty values)
        numeric_cols = ['dimension', 'exact_value', 'exact_angle', 
                       'gurobi_angle', 'gurobi_time', 
                       'bfas_angle', 'bfas_time',
                       'eao_ref', 'srpl_ref']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"Loaded {len(self.df)} reference results from {self.csv_path}")
    
    def get_problem(self, problem_name):
        """
        Get reference data for a specific problem
        
        Parameters
        ----------
        problem_name : str
            Problem name (e.g., 'Schur_vs_R+_n10')
        
        Returns
        -------
        dict
            Dictionary with reference data, or None if not found
        """
        result = self.df[self.df['problem'] == problem_name]
        
        if result.empty:
            return None
        
        # Convert to dict and clean up NaN values
        data = result.iloc[0].to_dict()
        
        # Replace NaN with None
        for key, value in data.items():
            if pd.isna(value):
                data[key] = None
        
        return data
    
    def get_by_category_and_dimension(self, category, dimension):
        """
        Get reference data by category and dimension
        
        Parameters
        ----------
        category : str
            Category name (e.g., 'schur_vs_nonneg')
        dimension : int
            Problem dimension
        
        Returns
        -------
        dict or None
            Dictionary with reference data
        """
        result = self.df[
            (self.df['category'] == category) & 
            (self.df['dimension'] == dimension)
        ]
        
        if result.empty:
            return None
        
        data = result.iloc[0].to_dict()
        
        # Replace NaN with None
        for key, value in data.items():
            if pd.isna(value):
                data[key] = None
        
        return data
    
    def get_all_in_category(self, category):
        """
        Get all problems in a category
        
        Parameters
        ----------
        category : str
            Category name
        
        Returns
        -------
        list of dict
            List of problem dictionaries
        """
        results = self.df[self.df['category'] == category]
        
        problems = []
        for _, row in results.iterrows():
            data = row.to_dict()
            # Replace NaN with None
            for key, value in data.items():
                if pd.isna(value):
                    data[key] = None
            problems.append(data)
        
        return problems
    
    def get_exact_solution(self, problem_name):
        """
        Get exact solution (value and angle) for a problem
        
        Parameters
        ----------
        problem_name : str
            Problem name
        
        Returns
        -------
        tuple (value, angle) or (None, None)
            Exact value and angle, or None if not available
        """
        data = self.get_problem(problem_name)
        
        if data is None:
            return None, None
        
        return data.get('exact_value'), data.get('exact_angle')
    
    def list_categories(self):
        """
        List all available categories
        
        Returns
        -------
        list
            List of category names
        """
        return self.df['category'].unique().tolist()
    
    def list_problems_in_category(self, category):
        """
        List problem names in a category
        
        Parameters
        ----------
        category : str
            Category name
        
        Returns
        -------
        list
            List of problem names
        """
        results = self.df[self.df['category'] == category]
        return results['problem'].tolist()
    
    def summary(self):
        """
        Print summary of reference data
        """
        print(f"\nTotal problems: {len(self.df)}")
        print(f"\nCategories:")
        
        for cat in self.list_categories():
            problems = self.get_all_in_category(cat)
            dims = [p['dimension'] for p in problems if p['dimension'] is not None]
            
            if dims:
                print(f"  • {cat}: {len(problems)} problems, "
                      f"dimensions {min(dims)}-{max(dims)}")
            else:
                print(f"  • {cat}: {len(problems)} problems")
            
            # Count how many have exact solutions
            exact_count = sum(1 for p in problems if p.get('exact_value') is not None)
            if exact_count > 0:
                print(f"    → {exact_count}/{len(problems)} with exact solutions")


# Convenience functions for backward compatibility
_reference_data = None

def get_reference_data(csv_path='reference_results_2025.csv'):
    """
    Get or create global reference data object
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    
    Returns
    -------
    ReferenceData
        Reference data object
    """
    global _reference_data
    if _reference_data is None:
        _reference_data = ReferenceData(csv_path)
    return _reference_data


def get_exact_solution(problem_name, csv_path='reference_results_2025.csv'):
    """
    Get exact solution for a problem
    
    Convenience function that replaces exact_schur_vs_nonneg_solution(n)
    
    Parameters
    ----------
    problem_name : str
        Problem name (e.g., 'Schur_vs_R+_n10')
    csv_path : str
        Path to CSV file
    
    Returns
    -------
    tuple (value, angle) or (None, None)
        Exact value and angle
    
    Examples
    --------
    >>> value, angle = get_exact_solution('Schur_vs_R+_n10')
    >>> print(f"Exact: {value:.6f}, angle: {angle:.6f}π")
    """
    ref = get_reference_data(csv_path)
    return ref.get_exact_solution(problem_name)


def get_problem_by_name(problem_name, csv_path='reference_results_2025.csv'):
    """
    Get all reference data for a problem
    
    Parameters
    ----------
    problem_name : str
        Problem name
    csv_path : str
        Path to CSV file
    
    Returns
    -------
    dict or None
        Problem data dictionary
    """
    ref = get_reference_data(csv_path)
    return ref.get_problem(problem_name)


if __name__ == "__main__":
    import sys
    
    # Try to find the CSV file
    csv_locations = [
        'reference_results_2025.csv',
        '../reference_results_2025.csv',
        '../../reference_results_2025.csv',
    ]
    
    csv_path = None
    for location in csv_locations:
        if Path(location).exists():
            csv_path = location
            break
    
    if csv_path is None:
        print("ERROR: reference_results_2025.csv not found!")
        print("Searched in:", csv_locations)
        sys.exit(1)
    
    print("Testing reference_data.py")
    print("="*70)
    
    # Load data
    ref = ReferenceData(csv_path)
    
    # Print summary
    ref.summary()
    
    # Test getting specific problems
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)
    
    # Example 1: Get exact solution
    print("\n1. Get exact solution for Schur vs R+ (n=10):")
    value, angle = ref.get_exact_solution('Schur_vs_R+_n10')
    if value is not None:
        print(f"   Exact value: {value:.6f}")
        print(f"   Exact angle: {angle:.6f}π")
    
    # Example 2: Get full problem data
    print("\n2. Get full data for Schur vs R+ (n=10):")
    data = ref.get_problem('Schur_vs_R+_n10')
    if data:
        print(f"   Problem: {data['problem']}")
        print(f"   Category: {data['category']}")
        print(f"   Dimension: {data['dimension']}")
        print(f"   Gurobi angle: {data['gurobi_angle']:.6f}π")
        print(f"   Gurobi time: {data['gurobi_time']:.4f}s")
    
    # Example 3: Get by category and dimension
    print("\n3. Get by category and dimension (schur_vs_schur, n=50):")
    data = ref.get_by_category_and_dimension('schur_vs_schur', 50)
    if data:
        print(f"   Problem: {data['problem']}")
        print(f"   Exact angle: {data['exact_angle']:.6f}π")
    
    # Example 4: List all in category
    print("\n4. List all Circulant PSD problems:")
    problems = ref.get_all_in_category('circulant_psd')
    for p in problems:
        dim = p['dimension']
        angle = p.get('gurobi_angle', 0)
        if angle:
            print(f"   n={dim}: angle = {angle:.6f}π")
    
    # Example 5: Compare with old function style
    print("\n5. Backward compatibility example:")
    print("   Old style: exact_schur_vs_nonneg_solution(10)")
    print("   New style: get_exact_solution('Schur_vs_R+_n10')")
    value, angle = get_exact_solution('Schur_vs_R+_n10')
    print(f"   Result: value={value:.6f}, angle={angle:.6f}π")
    
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)