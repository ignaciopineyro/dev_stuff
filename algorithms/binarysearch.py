from typing import List


# Binary Search - https://leetcode.com/problems/binary-search/description/
# Time Complexity: O(log n)
# Space Complexity: O(1)
def search(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# Search in Rotated Sorted Array - https://leetcode.com/problems/search-in-rotated-sorted-array/description/?envType=problem-list-v2&envId=binary-search
# Time Complexity: O(log n)
# Space Complexity: O(1)
def search_rotated(self, nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1

        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


# Find First and Last Position of Element in Sorted Array - https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=problem-list-v2&envId=binary-search
# Time Complexity: O(log n)
# Space Complexity: O(1)
def search_range(nums: List[int], target: int) -> List[int]:
    def search(x):
        lo, hi = 0, len(nums)
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    lo = search(target)
    hi = search(target + 1) - 1

    if lo <= hi:
        return [lo, hi]

    return [-1, -1]
