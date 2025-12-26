# Maximum Average Subarray I - https://leetcode.com/problems/maximum-average-subarray-i/description/?envType=problem-list-v2&envId=sliding-window
# Time Complexity: O(n)
# Space Complexity: O(1)
from typing import List


def findMaxAverage(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: float
    """
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, window_sum)

    return max_sum / k


# Longest Substring Without Repeating Characters - https://leetcode.com/problems/longest-substring-without-repeating-characters/?envType=problem-list-v2&envId=sliding-window
# Time Complexity: O(n)
# Space Complexity: O(min(m, n)) where m is the size of the charset
def lengthOfLongestSubstring(self, s):
    """
    :type s: str
    :rtype: int
    """
    seen = set()
    left = 0
    max_len = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1

        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len


# Minimum Operations to Make Binary Array Elements Equal to One I - https://leetcode.com/problems/minimum-operations-to-make-binary-array-elements-equal-to-one-i/description/
# Time Complexity: O(n)
# Space Complexity: O(1)
def minOperations(nums: List[int]) -> int:
    result = 0

    for i in range(len(nums) - 2):
        if nums[i] == 0:
            nums[i] ^= 1
            nums[i + 1] ^= 1
            nums[i + 2] ^= 1
            result += 1

    if not (nums[len(nums) - 3] == nums[len(nums) - 2] == nums[len(nums) - 1]):
        return -1

    return result
