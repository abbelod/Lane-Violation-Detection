"""
Safety-Critical Line Post-Processing for Lane Detection
Merges lines with similar slopes from Hough Transform output

SAFETY WARNING: This is for educational/demonstration purposes.
Production safety-critical systems require extensive validation,
testing, redundancy, and regulatory compliance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


class LinePostProcessor:
    """
    Post-process Hough Transform lines with safety-critical features.
    
    Merges lines with similar slopes and positions to create unified
    lane line representations, essential for accurate lane detection.
    """
    
    def __init__(self, 
                 slope_threshold: float = 0.1,
                 distance_threshold: float = 50.0,
                 min_line_length: float = 30.0,
                 max_merge_gap: float = 100.0):
        """
        Initialize line post-processor.
        
        Args:
            slope_threshold: Maximum slope difference for merging (radians)
            distance_threshold: Maximum distance between lines for merging (pixels)
            min_line_length: Minimum line length to keep (pixels)
            max_merge_gap: Maximum gap to bridge when merging (pixels)
        
        Safety: All parameters are validated and bounded.
        """
        # Validate parameters
        self.slope_threshold = self._validate_param(
            slope_threshold, 0.01, 1.0, "slope_threshold"
        )
        self.distance_threshold = self._validate_param(
            distance_threshold, 10.0, 200.0, "distance_threshold"
        )
        self.min_line_length = self._validate_param(
            min_line_length, 10.0, 500.0, "min_line_length"
        )
        self.max_merge_gap = self._validate_param(
            max_merge_gap, 20.0, 500.0, "max_merge_gap"
        )
        
        # Safety monitoring
        self.processing_errors = []
        self.last_input_count = 0
        self.last_output_count = 0
        self.merge_operations = 0
        
    def _validate_param(self, value: float, min_val: float, 
                       max_val: float, name: str) -> float:
        """
        Validate and bound parameter values for safety.
        
        Args:
            value: Parameter value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Parameter name for error messages
            
        Returns:
            Validated and bounded value
        """
        if not isinstance(value, (int, float)):
            warnings.warn(f"{name} must be numeric, using default")
            return (min_val + max_val) / 2
        
        if np.isnan(value) or np.isinf(value):
            warnings.warn(f"{name} is NaN or Inf, using default")
            return (min_val + max_val) / 2
        
        if value < min_val:
            warnings.warn(f"{name} too small, clamping to {min_val}")
            return min_val
        
        if value > max_val:
            warnings.warn(f"{name} too large, clamping to {max_val}")
            return max_val
        
        return float(value)
    
    def _validate_input_lines(self, lines: np.ndarray) -> bool:
        """
        Validate input lines array for safety.
        
        Args:
            lines: Lines array from cv2.HoughLinesP
            
        Returns:
            True if valid, False otherwise
        """
        self.processing_errors = []
        
        # Check for None
        if lines is None:
            self.processing_errors.append("NULL_INPUT")
            return False
        
        # Check for empty
        if lines.size == 0:
            self.processing_errors.append("EMPTY_INPUT")
            return False
        
        # Check dimensions
        if len(lines.shape) != 3:
            self.processing_errors.append(
                f"INVALID_DIMENSIONS: Expected 3D array, got {len(lines.shape)}D"
            )
            return False
        
        # Check last dimension is 4 (x1, y1, x2, y2)
        if lines.shape[2] != 4:
            self.processing_errors.append(
                f"INVALID_LINE_FORMAT: Expected 4 coordinates, got {lines.shape[2]}"
            )
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(lines)) or np.any(np.isinf(lines)):
            self.processing_errors.append("INVALID_VALUES: NaN or Inf detected")
            return False
        
        return True
    
    def _calculate_line_properties(self, x1: float, y1: float, 
                                   x2: float, y2: float) -> Dict:
        """
        Calculate line properties safely.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            
        Returns:
            Dictionary with line properties
        """
        # Calculate length
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        # Handle vertical lines (avoid division by zero)
        if abs(dx) < 1e-6:
            slope = np.inf if dy > 0 else -np.inf
            angle = np.pi / 2 if dy > 0 else -np.pi / 2
            intercept = x1  # For vertical lines, intercept is x-coordinate
        else:
            slope = dy / dx
            angle = np.arctan2(dy, dx)
            intercept = y1 - slope * x1
        
        # Midpoint
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        
        return {
            'x1': x1, 'y1': y1,
            'x2': x2, 'y2': y2,
            'length': length,
            'slope': slope,
            'angle': angle,
            'intercept': intercept,
            'mid_x': mid_x,
            'mid_y': mid_y
        }
    
    def _are_lines_similar(self, line1: Dict, line2: Dict) -> Tuple[bool, float]:
        """
        Check if two lines are similar enough to merge.
        
        Args:
            line1: First line properties
            line2: Second line properties
            
        Returns:
            (are_similar, distance_score)
        """
        # Check slope similarity
        # Handle infinite slopes (vertical lines)
        if np.isinf(line1['slope']) and np.isinf(line2['slope']):
            slope_similar = True
            angle_diff = 0.0
        elif np.isinf(line1['slope']) or np.isinf(line2['slope']):
            slope_similar = False
            angle_diff = np.pi / 2
        else:
            angle_diff = abs(line1['angle'] - line2['angle'])
            # Handle angle wraparound (-pi to pi)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            slope_similar = angle_diff < self.slope_threshold
        
        if not slope_similar:
            return False, float('inf')
        
        # Calculate perpendicular distance between lines
        # This measures how far apart the lines are (not their endpoints)
        
        if np.isinf(line1['slope']):
            # Both vertical lines
            distance = abs(line1['intercept'] - line2['intercept'])
        else:
            # Calculate perpendicular distance from midpoint of line2 to line1
            # Line equation: ax + by + c = 0
            # From y = mx + b -> mx - y + b = 0
            m = line1['slope']
            b = line1['intercept']
            
            # Distance from point (x0, y0) to line mx - y + b = 0
            # d = |mx0 - y0 + b| / sqrt(m^2 + 1)
            x0, y0 = line2['mid_x'], line2['mid_y']
            distance = abs(m * x0 - y0 + b) / np.sqrt(m**2 + 1)
        
        position_similar = distance < self.distance_threshold
        
        # Check gap between line segments
        # Calculate minimum distance between endpoints
        points1 = [(line1['x1'], line1['y1']), (line1['x2'], line1['y2'])]
        points2 = [(line2['x1'], line2['y1']), (line2['x2'], line2['y2'])]
        
        min_gap = float('inf')
        for p1 in points1:
            for p2 in points2:
                gap = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_gap = min(min_gap, gap)
        
        gap_acceptable = min_gap < self.max_merge_gap
        
        return (slope_similar and position_similar and gap_acceptable), distance
    
    def _merge_two_lines(self, line1: Dict, line2: Dict) -> Dict:
        """
        Merge two similar lines into one extended line.
        
        Args:
            line1: First line properties
            line2: Second line properties
            
        Returns:
            Merged line properties
        """
        # Collect all four endpoints
        points = [
            (line1['x1'], line1['y1']),
            (line1['x2'], line1['y2']),
            (line2['x1'], line2['y1']),
            (line2['x2'], line2['y2'])
        ]
        
        # Find the two endpoints that are farthest apart
        max_dist = 0
        best_pair = (points[0], points[1])
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.sqrt(
                    (points[i][0] - points[j][0])**2 + 
                    (points[i][1] - points[j][1])**2
                )
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (points[i], points[j])
        
        # Create merged line from farthest endpoints
        x1, y1 = best_pair[0]
        x2, y2 = best_pair[1]
        
        return self._calculate_line_properties(x1, y1, x2, y2)
    
    def _cluster_and_merge_lines(self, line_properties: List[Dict]) -> List[Dict]:
        """
        Cluster similar lines and merge them using union-find algorithm.
        
        Args:
            line_properties: List of line property dictionaries
            
        Returns:
            List of merged line properties
        """
        n = len(line_properties)
        if n == 0:
            return []
        
        # Union-Find data structure for clustering
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
        
        # Find all pairs of similar lines and union them
        for i in range(n):
            for j in range(i + 1, n):
                are_similar, _ = self._are_lines_similar(
                    line_properties[i], 
                    line_properties[j]
                )
                if are_similar:
                    union(i, j)
                    self.merge_operations += 1
        
        # Group lines by cluster
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(line_properties[i])
        
        # Merge lines within each cluster
        merged_lines = []
        for cluster in clusters.values():
            # Iteratively merge all lines in cluster
            merged = cluster[0]
            for line in cluster[1:]:
                merged = self._merge_two_lines(merged, line)
            merged_lines.append(merged)
        
        return merged_lines
    
    def _filter_short_lines(self, line_properties: List[Dict]) -> List[Dict]:
        """
        Filter out lines shorter than minimum length.
        
        Args:
            line_properties: List of line property dictionaries
            
        Returns:
            Filtered list of lines
        """
        return [
            line for line in line_properties 
            if line['length'] >= self.min_line_length
        ]
    
    def post_process_lines(self, lines: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Post-process Hough transform lines with merging and filtering.
        
        Args:
            lines: Output from cv2.HoughLinesP with shape (N, 1, 4)
                   where each line is [x1, y1, x2, y2]
        
        Returns:
            (processed_lines, processing_info)
            
            processed_lines: Merged and filtered lines in same format as input
            processing_info: Dictionary with:
                - status: Processing status string
                - input_count: Number of input lines
                - output_count: Number of output lines
                - merge_count: Number of merge operations
                - filtered_count: Number of lines filtered out
                - errors: List of errors/warnings
                - confidence: Processing confidence [0-1]
        """
        # Reset monitoring
        self.processing_errors = []
        self.merge_operations = 0
        
        # Safety: Validate input
        if not self._validate_input_lines(lines):
            return None, {
                'status': 'FAILED',
                'input_count': 0,
                'output_count': 0,
                'merge_count': 0,
                'filtered_count': 0,
                'errors': self.processing_errors,
                'confidence': 0.0
            }
        
        try:
            # Track input count
            self.last_input_count = lines.shape[0]
            
            # Extract line coordinates and calculate properties
            line_properties = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                props = self._calculate_line_properties(x1, y1, x2, y2)
                line_properties.append(props)
            
            # Filter out very short lines first (noise reduction)
            line_properties = self._filter_short_lines(line_properties)
            filtered_count = self.last_input_count - len(line_properties)
            
            # Cluster and merge similar lines
            merged_properties = self._cluster_and_merge_lines(line_properties)
            
            # Safety check: Ensure we have valid output
            if not merged_properties:
                return None, {
                    'status': 'NO_LINES_REMAINING',
                    'input_count': self.last_input_count,
                    'output_count': 0,
                    'merge_count': self.merge_operations,
                    'filtered_count': filtered_count,
                    'errors': ['All lines filtered or merged to nothing'],
                    'confidence': 0.0
                }
            
            # Convert back to numpy array format
            output_lines = np.zeros((len(merged_properties), 1, 4), dtype=np.float32)
            for i, line in enumerate(merged_properties):
                output_lines[i, 0] = [line['x1'], line['y1'], 
                                     line['x2'], line['y2']]
            
            # Track output count
            self.last_output_count = len(merged_properties)
            
            # Calculate confidence based on reduction ratio
            reduction_ratio = self.last_output_count / max(self.last_input_count, 1)
            confidence = min(1.0, max(0.0, 1.0 - reduction_ratio + 0.5))
            
            # Determine status
            if self.last_output_count >= self.last_input_count * 0.8:
                status = 'MINIMAL_MERGING'
            elif self.last_output_count >= self.last_input_count * 0.3:
                status = 'SUCCESS'
            else:
                status = 'AGGRESSIVE_MERGING'
                self.processing_errors.append(
                    'High merge ratio - verify parameters'
                )
            
            return output_lines, {
                'status': status,
                'input_count': self.last_input_count,
                'output_count': self.last_output_count,
                'merge_count': self.merge_operations,
                'filtered_count': filtered_count,
                'errors': self.processing_errors,
                'confidence': confidence,
                'reduction_ratio': reduction_ratio
            }
            
        except Exception as e:
            self.processing_errors.append(f'EXCEPTION: {str(e)}')
            return None, {
                'status': 'EXCEPTION',
                'input_count': self.last_input_count,
                'output_count': 0,
                'merge_count': self.merge_operations,
                'filtered_count': 0,
                'errors': self.processing_errors,
                'confidence': 0.0
            }
    
    def get_processing_stats(self) -> Dict:
        """
        Get processing statistics for safety monitoring.
        
        Returns:
            Statistics dictionary
        """
        return {
            'last_input_count': self.last_input_count,
            'last_output_count': self.last_output_count,
            'last_merge_count': self.merge_operations,
            'reduction_ratio': (self.last_output_count / max(self.last_input_count, 1)),
            'recent_errors': self.processing_errors,
            'parameters': {
                'slope_threshold': self.slope_threshold,
                'distance_threshold': self.distance_threshold,
                'min_line_length': self.min_line_length,
                'max_merge_gap': self.max_merge_gap
            }
        }


# ============================================================================
# CONVENIENCE FUNCTION - Direct usage without class
# ============================================================================

def post_process_lines(lines: np.ndarray,
                       slope_threshold: float = 0.1,
                       distance_threshold: float = 50.0,
                       min_line_length: float = 30.0,
                       max_merge_gap: float = 100.0) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Convenience function to post-process Hough transform lines.
    
    Merges lines with similar slopes and positions into unified lines.
    Essential for lane detection to combine fragmented line segments.
    
    Args:
        lines: Output from cv2.HoughLinesP, shape (N, 1, 4)
               Each line is [x1, y1, x2, y2]
        slope_threshold: Max slope difference for merging (radians, default 0.1)
        distance_threshold: Max distance between lines (pixels, default 50)
        min_line_length: Min line length to keep (pixels, default 30)
        max_merge_gap: Max gap to bridge (pixels, default 100)
    
    Returns:
        (merged_lines, info_dict)
        
        merged_lines: Processed lines array or None if failed
        info_dict: Processing information including status, counts, errors
    
    Example:
        >>> lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
        ...                         minLineLength=30, maxLineGap=10)
        >>> merged_lines, info = post_process_lines(lines)
        >>> print(f"Merged {info['input_count']} -> {info['output_count']} lines")
    
    Safety Features:
        - Input validation (null, dimension, value checks)
        - Parameter bounds validation
        - Exception handling
        - Status reporting
        - Confidence metrics
    """
    processor = LinePostProcessor(
        slope_threshold=slope_threshold,
        distance_threshold=distance_threshold,
        min_line_length=min_line_length,
        max_merge_gap=max_merge_gap
    )
    
    return processor.post_process_lines(lines)


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def demonstrate_line_merging():
    """Demonstrate the line post-processing with visualizations."""
    import cv2
    
    print("="*70)
    print("LINE POST-PROCESSING DEMONSTRATION")
    print("Safety-Critical Line Merging for Lane Detection")
    print("="*70)
    print()
    
    # Create test image with fragmented lines
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Simulate fragmented lane lines (what Hough might return)
    # Left lane - multiple segments
    cv2.line(test_img, (100, 400), (150, 300), (255, 255, 255), 2)
    cv2.line(test_img, (160, 280), (200, 200), (255, 255, 255), 2)
    cv2.line(test_img, (210, 180), (250, 100), (255, 255, 255), 2)
    
    # Right lane - multiple segments
    cv2.line(test_img, (500, 400), (450, 300), (255, 255, 255), 2)
    cv2.line(test_img, (440, 280), (400, 200), (255, 255, 255), 2)
    cv2.line(test_img, (390, 180), (350, 100), (255, 255, 255), 2)
    
    # Some noise lines
    cv2.line(test_img, (300, 350), (320, 320), (255, 255, 255), 2)
    cv2.line(test_img, (280, 200), (290, 190), (255, 255, 255), 2)
    
    print("✓ Created test image with fragmented lines")
    
    # Detect lines with Hough
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=20, maxLineGap=5)
    
    if lines is None:
        print("✗ No lines detected")
        return
    
    print(f"✓ Hough detected {len(lines)} line segments")
    print()
    
    # Draw original lines
    original_img = test_img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(original_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                (0, 0, 255), 2)
    
    # Post-process lines
    print("Processing lines...")
    processor = LinePostProcessor(
        slope_threshold=0.15,      # ~8.6 degrees
        distance_threshold=50.0,   # 50 pixels
        min_line_length=30.0,      # 30 pixels minimum
        max_merge_gap=100.0        # Bridge 100 pixel gaps
    )
    
    merged_lines, info = processor.post_process_lines(lines)
    
    # Display results
    print("="*70)
    print("PROCESSING RESULTS:")
    print("="*70)
    print(f"Status: {info['status']}")
    print(f"Input lines: {info['input_count']}")
    print(f"Output lines: {info['output_count']}")
    print(f"Merge operations: {info['merge_count']}")
    print(f"Filtered out: {info['filtered_count']}")
    print(f"Reduction ratio: {info['reduction_ratio']:.2%}")
    print(f"Confidence: {info['confidence']:.2%}")
    
    if info['errors']:
        print(f"\nWarnings/Errors:")
        for error in info['errors']:
            print(f"  - {error}")
    print()
    
    # Draw merged lines
    if merged_lines is not None:
        merged_img = test_img.copy()
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(merged_img, (int(x1), int(y1)), (int(x2), int(y2)), 
                    (0, 255, 0), 3)
        
        # Save results
        cv2.imwrite('/home/claude/original_lines.jpg', original_img)
        cv2.imwrite('/home/claude/merged_lines.jpg', merged_img)
        
        # Create side-by-side comparison
        comparison = np.hstack([original_img, merged_img])
        cv2.putText(comparison, "BEFORE: Fragmented", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "AFTER: Merged", (610, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite('/home/claude/comparison.jpg', comparison)
        
        print("✓ Results saved:")
        print("  - original_lines.jpg (red = fragmented lines)")
        print("  - merged_lines.jpg (green = merged lines)")
        print("  - comparison.jpg (side-by-side)")
    else:
        print("✗ Post-processing failed")
    
    # Display statistics
    print()
    print("="*70)
    print("PROCESSING STATISTICS:")
    print("="*70)
    stats = processor.get_processing_stats()
    print(f"Lines reduced: {stats['last_input_count']} → {stats['last_output_count']}")
    print(f"Merge operations: {stats['last_merge_count']}")
    print(f"Reduction: {(1 - stats['reduction_ratio']) * 100:.1f}%")
    print()
    print("Parameters used:")
    for key, value in stats['parameters'].items():
        print(f"  {key}: {value}")
    
    print()
    print("="*70)
    print("SAFETY REMINDER:")
    print("Always validate post-processing results in safety-critical systems.")
    print("Monitor merge statistics and confidence metrics continuously.")
    print("="*70)


if __name__ == "__main__":
    demonstrate_line_merging()
