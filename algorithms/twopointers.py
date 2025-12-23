# Count pairs whose sum is less than target - https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/description/?envType=problem-list-v2&envId=two-pointers
# Time complexity: O(n log n) due to sorting
# Space complexity: O(1) if we ignore the space used by sorting
def countPairs(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    nums.sort()
    lp = 0
    rp = len(nums) - 1
    result = 0

    while lp < rp:
        sum = nums[lp] + nums[rp]

        if sum < target:
            result += rp - lp
            lp += 1
        else:
            rp -= 1

    return result


# Reverse prefix of word - https://leetcode.com/problems/reverse-prefix-of-word/description/?envType=problem-list-v2&envId=two-pointers
# Time complexity: O(n) where n is the length of the input string
# Space complexity: O(n) because of the new list (strings are immutable in Python)
def reversePrefix(word, ch):
    """
    :type word: str
    :type ch: str
    :rtype: str
    """
    match_index = None
    for i in range(len(word)):
        if word[i] == ch:
            match_index = i
            break

    if not match_index:
        return word

    lp, rp = 0, match_index
    new_word = list(word)
    while lp < rp:
        new_word[lp], new_word[rp] = word[rp], word[lp]
        lp += 1
        rp -= 1
    return "".join(new_word)


# Container with most water - https://leetcode.com/problems/container-with-most-water/description/
# Time complexity: O(n)
# Space complexity: O(1)
def maxArea(height):
    """
    :type height: List[int]
    :rtype: int
    """
    area = 0
    lp = 0
    rp = len(height) - 1
    while lp < rp:
        new_area = min(height[lp], height[rp]) * (rp - lp)
        if new_area > area:
            area = new_area
        if height[lp] > height[rp]:
            rp -= 1
        else:
            lp += 1
    return area


# Linked List Cycle - https://leetcode.com/problems/linked-list-cycle/description/
# Time complexity: O(n)
# Space complexity: O(1)
def hasCycle(head):
    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    """
    :type head: ListNode
    :rtype: bool
    """
    sp, fp = head, head
    while fp is not None and fp.next is not None:
        sp = sp.next
        fp = fp.next.next

        if sp is fp:
            return True

    return False


# Middle of the Linked List - https://leetcode.com/problems/middle-of-the-linked-list/description/
# Time complexity: O(n)
# Space complexity: O(1)
def middleNode(self, head):
    # Definition for singly-linked list.
    # class ListNode(object):
    #     def __init__(self, val=0, next=None):
    #         self.val = val
    #         self.next = next
    """
    :type head: Optional[ListNode]
    :rtype: Optional[ListNode]
    """
    sp, fp = head, head

    while fp is not None and fp.next is not None:
        sp = sp.next
        fp = fp.next.next

    return sp
