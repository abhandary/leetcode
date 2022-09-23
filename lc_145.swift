
class LRUCache {

    class DLinkedNode {
        var key: Int
        var value: Int
        var prev: DLinkedNode?
        var next: DLinkedNode?
        
        init(_ key: Int, _ value: Int) {
            self.key = key
            self.value = value
        }
    }
    
    func add(node: DLinkedNode) {
        // always add at head
        node.next = head.next
        node.next?.prev = node            
            
        head.next = node
        node.prev = head
    }
        
    func remove(node: DLinkedNode) {
        let prev = node.prev
        let next = node.next
            
        next?.prev = prev
        prev?.next = next
    }
        
    func moveToHead(node: DLinkedNode) {
        remove(node: node)
        add(node: node)
    }
    
    func popTail() -> DLinkedNode? {
        if let res = tail.prev {
            remove(node: res)
            return res
        }
        return nil
    }
    
    var capacity: Int
    var head: DLinkedNode
    var tail: DLinkedNode
    var cache: [Int : DLinkedNode] = [:]
    
    init(_ capacity: Int) {
        self.capacity = capacity
        head = DLinkedNode(-1, -1)
        tail = DLinkedNode(-1, -1)
        head.next = tail
        tail.prev = head
    }
    
    func get(_ key: Int) -> Int {
        if let storedNode = cache[key] {
            moveToHead(node: storedNode)
            return storedNode.value
        }
        return -1
    }
    
    func put(_ key: Int, _ value: Int) {
        if let storedNode = cache[key] {
            moveToHead(node: storedNode)
            storedNode.value = value
        } else {
            let newNode = DLinkedNode(key, value)
            add(node: newNode)
            cache[key] = newNode
            if cache.count > self.capacity {
                if let tail = popTail() {
                    cache[tail.key] = nil
                }
            }
        }
    }
}

    
    // LC1. Two Sum
    // https://leetcode.com/problems/two-sum/
    // O(N), O(N)
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        var dict: [Int:Int] = [:]
        for (ix, value) in nums.enumerated() {
            if let storedIx = dict[target - value] {
                return [storedIx, ix]
            }
            dict[value] = ix
        }
        return []
    }

    // LC2. Add Two Numbers
    // https://leetcode.com/problems/add-two-numbers/
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        let dummyHead = ListNode()
        var itr: ListNode? = nil
        var l1Itr = l1
        var l2Itr = l2
        dummyHead.next = itr
        var carry = 0
        while (l1Itr != nil || l2Itr != nil) {
            var sum = 0
            if let l1Val = l1Itr?.val {
                sum += l1Val
            }
            if let l2Val = l2Itr?.val {
                sum += l2Val
            }
            sum += carry
            let newNode = ListNode(sum % 10)
            carry = sum / 10
            itr?.next = newNode
            itr = newNode
            l1Itr = l1Itr?.next
            l2Itr = l2Itr?.next
            if dummyHead.next == nil {
                dummyHead.next = newNode
            }
        }
        if carry > 0 {
            itr?.next = ListNode(carry)
        }
        return dummyHead.next
    }

    // LC3. Longest Substring Without Repeating Characters
    // https://leetcode.com/problems/longest-substring-without-repeating-characters/
    func lengthOfLongestSubstring(_ s: String) -> Int {
        var len = 0
        var startIx = 0
        var indexMap: [Character: Int] = [:] // can use a character set array instead
        for ix in 0..<s.count {
            let c = s[s.index(s.startIndex, offsetBy: ix)]
            if let storedIx = indexMap[c], storedIx >= startIx {
                startIx = storedIx + 1
            }
            len = max(len, ix - startIx + 1)            
            indexMap[c] = ix
        }
        return len
    }
    // LC4. Median of Two Sorted Arrays
    // https://leetcode.com/problems/median-of-two-sorted-arrays/
    // O(log(m+n))
    // O(1)
    // Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
    // Follow up: The overall run time complexity should be O(log (m+n)).
    func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        let m = nums1.count
        let n = nums2.count
        let total = m + n
        if total % 2 == 1 {
            return Double(findKth(nums1, nums2, total / 2 + 1))
        } else {
            return Double(findKth(nums1, nums2, total / 2) + findKth(nums1, nums2, total / 2 + 1)) / 2
        }
    }
    
    func findKth(_ nums1: [Int], _ nums2: [Int], _ k: Int) -> Int {
        if nums1.count == 0 {
            return nums2[k - 1]
        }
        if nums2.count == 0 {
            return nums1[k - 1]
        }
        if k == 1 {
            return min(nums1[0], nums2[0])
        }
        let mid1 = min(k / 2, nums1.count)
        let mid2 = min(k / 2, nums2.count)
        if nums1[mid1 - 1] < nums2[mid2 - 1] {
            return findKth(Array(nums1[mid1..<nums1.count]), nums2, k - mid1)
        } else {
            return findKth(nums1, Array(nums2[mid2..<nums2.count]), k - mid2)
        }
    }

    // LC5. Longest Palindromic Substring
    // https://leetcode.com/problems/longest-palindromic-substring/
    func expandAroundCenter(_ schars: [Character], _ ix: Int, _ jx: Int) -> Int {
        var ix = ix, jx = jx
        while ix >= 0 && jx < schars.count && schars[ix] == schars[jx] {
            ix -= 1
            jx += 1
        }
        return jx - ix - 1
    }
    
    func longestPalindrome(_ s: String) -> String {
        let schars = Array(s)
        var left = 0, right = 0
        var currentLongest = 0
        for ix in 0...s.count - 1 {
            let oddLength = expandAroundCenter(schars, ix, ix)
            let evenLength = expandAroundCenter(schars, ix, ix + 1)
            let maxLen = max(oddLength, evenLength)
            if maxLen > right - left {
                left = ix - (maxLen - 1) / 2   
                right = ix + maxLen / 2   
            }
        }
        return String(schars[left...right])
    }

    // LC7. Reverse Integer
    // https://leetcode.com/problems/reverse-integer/
    func reverse(_ x: Int) -> Int {
        var reverse:Int32 = 0
        var x = Int32(x)
        while x != 0 {
            let nextDigit = x % 10
            x /= 10            
            if reverse > Int32.max / 10 || reverse == Int32.max / 10 && nextDigit > 7 {
                return 0
            }
            if reverse < Int32.min / 10 || reverse == Int32.min / 10 && nextDigit < -8 {
                return 0
            }            
            reverse = reverse * 10 + nextDigit
        }
        return Int(reverse)
    }

    // not working alternative
     func reverse(_ x: Int) -> Int {
        var reverse:Int32 = 0
        var x = Int32(x)
        while x != 0 {
            let nextDigit = x % 10
            x /= 10            
            let nextValue = reverse * 10 + nextDigit
            if (nextValue - nextDigit) / 10 != reverse {
                return 0
            }
            reverse = nextValue
        }
        return Int(reverse)
    }

    // LC8. String to Integer (atoi)
    // https://leetcode.com/problems/string-to-integer-atoi/
    // Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).
    // fails for Input = "-91283472332"
    func myAtoi(_ s: String) -> Int {
        let schars = Array(s)
        var sign = 1, num = 0, i = 0
        while i < schars.count && schars[i] == " " {
            i += 1
        }
        if i < schars.count && (schars[i] == "-" || schars[i] == "+") {
            sign = schars[i] == "-" ? -1 : 1
            i += 1
        }

        while i < schars.count && schars[i] >= "0" && schars[i] <= "9" {
            let digit = Int(String(schars[i]))!
            if (num > Int32.max / 10) || (num == Int32.max / 10 && digit > Int32.max % 10) {
                return sign == 1 ? Int(Int32.max) : Int(Int32.min)
            }
            
            num = num * 10 + digit
            print(num)
            i += 1
        }
        return num * sign
    }

    // passes all cases
    func myAtoi(_ s: String) -> Int {
        let schars = Array(s)
        var result = Int32(0)
        var rx = 0
        while rx < schars.count && schars[rx] == " " {
            rx += 1
        }
        var isPositive = false
        var isNegative = false        
        if rx < schars.count && schars[rx] == "+" { rx += 1; isPositive = true }
        if rx < schars.count && schars[rx] == "-" { rx += 1; isNegative = true }
        
        if isPositive && isNegative { return Int(result) }
        
        while rx < schars.count {
            if !(schars[rx].isASCII && schars[rx].isNumber) {
                return Int(result)
            }
            if var digit = Int32(String(schars[rx])) {
                digit = isNegative ? -digit : digit
                print("\(result) \(digit) \(isNegative)")
                if (result > Int32.max / 10) || (result == Int32.max / 10 && digit > Int32.max % 10) {
                    return Int(Int32.max)
                }
                if (result < Int32.min / 10) || (result == Int32.min / 10 && digit < Int32.min % 10) {
                    return Int(Int32.min)
                }
                result = result * 10 + digit
            }
            rx += 1
        }
        return Int(result)
     }

    // LC10. Regular Expression Matching
    // https://leetcode.com/problems/regular-expression-matching/
    // https://www.youtube.com/watch?v=XZgjyL7qQqU
    // O(N), O(N)
    // Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
    // '.' Matches any single character.​​​​
    // '*' Matches zero or more of the preceding element.
    // The matching should cover the entire input string (not partial).
    func isMatch(_ s: String, _ p: String) -> Bool { // s = "aa", p = "a"
        let sChars = Array(s)
        let pChars = Array(p)
        var dp = Array(repeating: Array(repeating: false, count: pChars.count + 1), count: sChars.count + 1)
        dp[sChars.count][pChars.count] = true

        for i in stride(from: sChars.count, through: 0, by: -1) {
            for j in stride(from: pChars.count - 1, through: 0, by: -1) {
                let first_match = (i < sChars.count &&
                                       (pChars[j] == sChars[i] ||
                                        pChars[j] == "."));
                if (j + 1 < pChars.count && pChars[j+1] == "*"){
                    dp[i][j] = dp[i][j+2] || first_match && dp[i+1][j];
                } else {
                    dp[i][j] = first_match && dp[i+1][j+1];
                }
            }
        }
        return dp[0][0];
    }

    // recursion
    func isMatch(_ s: String, _ p: String) -> Bool { 
        if p.count == 0  {
            return s.count == 0
        }

        let firstCharMatches = s.count > 0 && (s.first == p.first || p.first == ".") 
    
        if p.count >= 2 && p[p.index(s.startIndex, offsetBy: 1)] == "*" {
            return isMatch(s, String(p.suffix(from: p.index(s.startIndex, offsetBy: 2)))) || (firstCharMatches && isMatch(String(s.suffix(from: s.index(s.startIndex, offsetBy: 1))), p))
        } else {
            return firstCharMatches && isMatch(String(s.suffix(from: s.index(s.startIndex, offsetBy: 1))), String(p.suffix(from: p.index(p.startIndex, offsetBy: 1))))
        }
    }

    // LC11. Container With Most Water
    // https://leetcode.com/problems/container-with-most-water/
    func maxArea(_ height: [Int]) -> Int {
        var left = 0, right = height.count - 1
        var maxArea = 0
        while left < right {
            let minHeight = min(height[left], height[right]) 
            let newArea = minHeight * (right - left)
            maxArea = max(maxArea, newArea)
            if height[left] < height[right] {
                left += 1
            } else {
                right -= 1
            }
        }
        return maxArea
    }

    // LC13. Roman to Integer
    // https://leetcode.com/problems/roman-to-integer/
    // Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
    func romanToInt(_ s: String) -> Int {
        let map : [Character : Int] = ["M" : 1000, "C" : 100, "D" : 500, "L" : 50, "X" : 10, "V" : 5, "I" : 1];
        var result = 0
        var i = 1
        var lastValue = 0
        while i <= s.count {
            let c = s[s.index(s.endIndex, offsetBy: -i)]
            if let currentValue = map[c] {
                if currentValue < lastValue {
                    result -= currentValue
                } else {
                    result += currentValue
                }
                lastValue = currentValue
            }
            i += 1
        }
        return result
    }

    // LC14. Longest Common Prefix
    // Write a function to find the longest common prefix string amongst an array of strings.
    // If there is no common prefix, return an empty string "".
    // Time complexity : O(S), where S is the sum of all characters in strings.
    // Space complexity : O(1).  
    func longestCommonPrefix(_ strs: [String]) -> String {
        guard strs.count > 0 else { return "" }
        var prefix = strs[0]
        for i in 1..<strs.count {
            while !strs[i].hasPrefix(prefix) {
                prefix.removeLast()
                if prefix.isEmpty {
                    return ""
                }
            }
        }
        return prefix
    }

    // vertical scanning
    // Time complexity : O(S), where S is the sum of all characters in strings.
    // Space complexity : O(1).
    func longestCommonPrefix(_ strs: [String]) -> String {
        guard strs.count > 0 else { return "" }
        for i in 0..<strs[0].count {
            let c = strs[0][strs[0].index(strs[0].startIndex, offsetBy: i)]
            for j in 1..<strs.count {
                if i == strs[j].count || c != strs[j][strs[j].index(strs[j].startIndex, offsetBy: i)] {
                    return String(strs[0][strs[0].startIndex..<strs[0].index(strs[0].startIndex, offsetBy: i)])
                }
            }
        }
        return strs[0]
    }

    // my solution
    func longestCommonPrefix(_ strs: [String]) -> String {
        var result = ""
        var ix = 0
        var currentChar: Character? = nil
        while (true) {
            for str in strs {
                if ix >= str.count {
                    return result
                }
                let char = str[str.index(str.startIndex, offsetBy: ix)]
                if currentChar == nil {
                    currentChar = char
                } else if currentChar != char {
                    return result
                }
            }
            if let currentChar = currentChar {
                result += String(currentChar)
            }
            ix += 1
            currentChar = nil
        }
        return  result
    }

    // divide and conquer and binary search also possible

    // passes 93 / 123 
    // fails Input: ["","b"]
    // Time complexity : preprocessing O(S)O(S), where SS is the number of all characters in the array, LCP query O(m)O(m).
    // Trie build has O(S)O(S) time complexity. To find the common prefix of qq in the Trie takes in the worst case O(m)O(m).
    // Space complexity : O(S)O(S). We only used additional SS extra space for the Trie.
    class TrieNode {
        var children: [Character: TrieNode] = [:]
        var isWord: Bool = false
        var word: String?

        func insert(_ word: String) {
            var node = self
            for char in word {
                if node.children[char] == nil {
                    node.children[char] = TrieNode()
                }
                node = node.children[char]!
            }
            node.isWord = true
            node.word = word
        }

        func longestCommonPrefix() -> String {
            var node = self
            var prefix = ""
            while node.children.count == 1 {
                let (key, _) = node.children.first!
                prefix.append(key)
                node = node.children[key]!
            }
            return prefix
        }
    }

    class Trie {
        var root: TrieNode = TrieNode()

        func insert(_ word: String) {
            root.insert(word)
        }

        func longestCommonPrefix() -> String {
            return root.longestCommonPrefix()
        }
    }
    
    // LC14. Longest Common Prefix
    func longestCommonPrefix(_ strs: [String]) -> String {
        guard strs.count > 0 else { return "" }
        if strs.count == 1 { return strs[0] }
        var trie = Trie()
        for str in strs {
            trie.insert(str)
        }
        return trie.longestCommonPrefix()
    }

    func longestCommonPrefix(_ strs: [String]) -> String {
        var result = ""
        var ix = 0
        var currentChar: Character? = nil
        while (true) {
            for str in strs {
                if ix >= str.count {
                    return result
                }
                let char = str[str.index(str.startIndex, offsetBy: ix)]
                if currentChar == nil {
                    currentChar = char
                } else if currentChar != char {
                    return result
                }
            }
            if let currentChar = currentChar {
                result += String(currentChar)
            }
            ix += 1
            currentChar = nil
        }
        return  result
    }

    func longestCommonPrefix(_ strs: [String]) -> String {
        var input: [[Character]] = []
        var maxLen = Int.min
        for str in strs {
            input.append(Array(str))
            maxLen = max(maxLen, str.count)
        }
        
        var result = ""
        var currentIx = 0
        while currentIx < maxLen {
            for (sx, schars) in input.enumerated() {
                if currentIx == schars.count { return result }
                if sx == 0 { continue }
                if schars[currentIx] != input[sx - 1][currentIx] {
                    return result
                }
            }
            result += String(input[0][currentIx])
            currentIx += 1
        }
        return  result
    }

    // LC15. 3Sum
    // Time Complexity: O(n^2) twoSumII is O(n), and we call it n times.
    // Space Complexity: from \mathcal{O}(\log{n})O(logn) to \mathcal{O}(n)O(n)
    func threeSum(_ nums: [Int]) -> [[Int]] {
        var result = [[Int]]()
        if nums.count < 3 { return result }
        let sortedNums = nums.sorted()
        for i in 0..<sortedNums.count - 2 {
            if i > 0 && sortedNums[i] == sortedNums[i - 1] { continue }
            var left = i + 1
            var right = sortedNums.count - 1
            while left < right {
                let sum = sortedNums[i] + sortedNums[left] + sortedNums[right]
                if sum == 0 {
                    result.append([sortedNums[i], sortedNums[left], sortedNums[right]])
                    left += 1
                    right -= 1
                    while left < right && sortedNums[left] == sortedNums[left - 1] { left += 1 }
                    while left < right && sortedNums[right] == sortedNums[right + 1] { right -= 1 }
                } else if sum < 0 {
                    left += 1
                } else {
                    right -= 1
                }
            }
        }
        return result
    }

    // my solution
    func threeSum(_ nums: [Int]) -> [[Int]] {
        let nums = nums.sorted { $0 < $1 }
        var result: [[Int]] = []
        guard nums.count >= 3 else { return result }
        for ix in 0...nums.count - 3 {
            if ix > 0 && nums[ix] == nums[ix - 1] { continue }
            twoSum(nums, ix + 1, nums.count - 1, nums[ix], &result)
        }
        return result
    }
    
    func twoSum(_ nums: [Int], _ left: Int, _ right: Int, _ initial: Int, _ result: inout [[Int]]) {
        var left = left
        var right = right
        while left < right {
            let sum = initial + nums[left] + nums[right]
            if sum == 0 {
                result.append([initial, nums[left], nums[right]]) 
                left += 1; right -= 1
                while left < right && nums[left] == nums[left - 1] { left += 1 }
            } else if sum < 0 {
                left += 1
            } else {
                right -= 1
            }
        }
    }


    // Time Complexity: O(n^2) twoSum is )O(n), and we call it n times.
    // Space Complexity: O(n) for the hashset.
    func threeSum(_ nums: [Int]) -> [[Int]] {
        var result = [[Int]]()
        if nums.count < 3 { return result }
        let sortedNums = nums.sorted()
        var i = 0
        while i < nums.count && nums[i] <= 0 {
            if (i == 0 || nums[i - 1] != nums[i]) {
                twoSum(sortedNums, i, &result)                
            }
            i += 1
        }
        return result
    }
    
    func twoSum(_ nums: [Int], _ i: Int, _ result: inout [[Int]]) {
        var j = i + 1
        var seen = Set<Int>()
        while j < nums.count {
            let target = -nums[i] - nums[j]
            if seen.contains(target) {
                result.append([nums[i], nums[j], target])
                while (j + 1 < nums.count && nums[j] == nums[j + 1]) { j += 1 }
            }
            seen.insert(nums[j])
            j += 1
        }
    }

    // LC17. Letter Combinations of a Phone Number
    // https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    // O? 
    func letterCombinationsHelper(_ map: [Character:String], _ dchars: [Character], _ ix: Int, _ partial: String, _  result: inout [String]) {
        if partial.count == dchars.count {
            result.append(partial)
            return
        }
        if let mapped = map[dchars[ix]] {
            let mchars = Array(mapped)
            for mchar in mchars {
                letterCombinationsHelper(map, dchars, ix + 1, partial + String(mchar), &result)
            }
        }
    }
    
    func letterCombinations(_ digits: String) -> [String] {
        guard digits.count > 0 else { return [] }
        let map: [Character : String] = ["1": "1", "2" : "abc", "3" : "def", "4" : "ghi", "5" : "jkl", "6" : "mno", "7" : "pqrs", "8" : "tuv", "9" : "wxyz"]    
        let dchars = Array(digits)
        var result: [String] = []
        letterCombinationsHelper(map, dchars, 0, "", &result)
        return result
    }

    // LC19. Remove Nth Node From End of List
    // Given a linked list, remove the n-th node from the end of list and return its head.
    // Time complexity : O(L)
    // Space complexity : O(1)
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? { 
        guard let head = head else { return nil }
        var dummy = ListNode(0)
        dummy.next = head
        var fast = dummy
        var slow = dummy
        for _ in 0..<n {
            fast = fast.next!
        }
        while fast.next != nil {
            fast = fast.next!
            slow = slow.next!
        }
        slow.next = slow.next?.next
        return dummy.next
    }

    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        guard head != nil else { return nil }
        let dummyHead = ListNode(0)
        dummyHead.next = head
        var itr = head
        var ix = 0
        while itr != nil && ix < n {
            itr = itr?.next
            ix += 1
        }
        if ix < n { return head }
        var trail: ListNode? = dummyHead
        while itr != nil {
            trail = trail?.next
            itr = itr?.next
        }
        trail?.next = trail?.next?.next
        return dummyHead.next
    }

    // LC20. Valid Parentheses
    // https://leetcode.com/problems/valid-parentheses/
    func isValid(_ s: String) -> Bool {
        var stack: [Character] = []
        let map: [Character: Character] = ["}": "{", "]": "[", ")": "("]
        for ix in 0..<s.count {
            let c = s[s.index(s.startIndex, offsetBy: ix)]
            if let mapped = map[c] {
                if stack.count == 0 { return false }
                if stack.removeLast() != mapped { return false }
            } else {
                stack.append(c)
            }
        }
        return stack.count == 0
    }

    // LC21. Merge Two Sorted Lists
    // Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
    // Return the head of the merged linked list.
    func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
        if list1 == nil {
            return list2
        }
        if list2 == nil {
            return list1
        }
        var head: ListNode?
        if list1!.val < list2!.val {
            head = list1
            head?.next = mergeTwoLists(list1?.next, list2)
        } else {
            head = list2
            head?.next = mergeTwoLists(list1, list2?.next)
        }
        return head
    }

    func mergeTwoLists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
       let dummy = ListNode(0)
        var p: ListNode? = dummy
        var l1 = list1
        var l2 = list2
        while l1 != nil && l2 != nil {
            if let l1Temp = l1, let l2Temp = l2, l1Temp.val < l2Temp.val {
                p?.next = l1
                l1 = l1?.next
            } else {
                p?.next = l2
                l2 = l2?.next
            }
            p = p?.next
        }
        if l1 != nil {
            p?.next = l1
        }
        if l2 != nil {
            p?.next = l2
        }
        return dummy.next
    }

    // LC22. Generate Parentheses
    func generateParenthesis(_ n: Int) -> [String] {
        var result = [String]()
        generateParenthesis(n, 0, 0, "", &result)
        return result
    }

    func generateParenthesis(_ n: Int, _ left: Int, _ right: Int, _ partial: String, _ result: inout [String]) {
        if left == n && right == n {
            result.append(partial)
            return
        }
        
        if left < n {
            generateParenthesis(n, left + 1, right, partial + "(", &result)
        }
        if right < left {
            generateParenthesis(n, left, right + 1, partial + ")", &result)
        }
    }

// LC23. Merge k Sorted Lists
public class _23 {
    public static class Solution1 {
        public ListNode mergeKLists(ListNode[] lists) {
            PriorityQueue<ListNode> heap = new PriorityQueue((Comparator<ListNode>) (o1, o2) -> o1.val - o2.val);

            for (ListNode node : lists) {
                if (node != null) {
                    heap.offer(node);
                }
            }

            ListNode pre = new ListNode(-1);
            ListNode temp = pre;
            while (!heap.isEmpty()) {
                ListNode curr = heap.poll();
                temp.next = new ListNode(curr.val);
                if (curr.next != null) {
                    heap.offer(curr.next);
                }
                temp = temp.next;
            }
            return pre.next;
        }
    }

}

    // LC26. Remove Duplicates from Sorted Array
    // Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.
    // relative order of the elements must not be changed.
    func removeDuplicates(_ nums: inout [Int]) -> Int { // O(n)
        if nums.count == 0 { return 0 }
        var i = 0
        for j in 1..<nums.count {
            if nums[i] != nums[j] {
                i += 1
                nums[i] = nums[j]
            }
        }
        return i + 1
    }

    func removeDuplicates(_ nums: inout [Int]) -> Int {
        guard nums.count > 0 else { return 0 }
        var wx = 1
        for rx in 1..<nums.count {
            if nums[rx] != nums[wx - 1] {
                nums[wx] = nums[rx]
                wx += 1
            }
        }
        return wx 
    }

    func removeDuplicates(_ nums: inout [Int]) -> Int {
        guard nums.count > 0 else { return 0 }
        var wx = 1
        var rx = 1
        while rx < nums.count {
            if nums[rx] != nums[rx - 1] {
                nums[wx] = nums[rx]
                wx += 1
            }
            rx += 1
        }
        return wx
    }

    // LC28. Implement strStr()
    func strStr(_ haystack: String, _ needle: String) -> Int {
        if needle.count == 0 { return 0 }
        if needle.count > haystack.count { return -1 }
        for i in 0..<haystack.count - needle.count + 1 {
            if haystack[i..<i+needle.count] == needle {
                return i
            }
        }
        return -1
    }

public class _28 {

  public static class Solution1 {
    public int strStr(String haystack, String needle) {
      if (haystack == null || needle == null || haystack.length() < needle.length()) {
        return -1;
      }

      for (int i = 0; i <= haystack.length() - needle.length(); i++) {
        if (haystack.substring(i, i + needle.length()).equals(needle)) {
          return i;
        }
      }
      return -1;
    }
  }

}

// LC29. Divide Two Integers
public class _29 {

    public static class Solution1 {
        /**
         * credit: https://leetcode.com/problems/divide-two-integers/solution/ solution 1
         * <p>
         * Key notes:
         * 1. dividend = Integer.MAX_VALUE and divisor = -1 is a special case which will be handled separately;
         * 2. because within the given range, [-2_31 to 2_31 - 1], every positive integer could be mapped to a corresponding negative integer while the opposite is not true
         * because of the smallest number: Integer.MIN_VALUE = -2147483648 doesn't have one (Integer.MAX_VALUE is 2147483647). So we'll turn both dividend and divisor into negative numbers to do the operation;
         * 3. division, in its essence, is subtraction multiple times until it cannot be subtracted any more
         * <p>
         * Time: O(n)
         * Space: O(1)
         */
        public int divide(int dividend, int divisor) {
            if (dividend == Integer.MIN_VALUE && divisor == -1) {
                return Integer.MAX_VALUE;
            }
            int negativeCount = 0;
            if (dividend < 0) {
                negativeCount++;
            } else {
                dividend = -dividend;
            }
            if (divisor < 0) {
                negativeCount++;
            } else {
                divisor = -divisor;
            }

            int quotient = 0;
            while (dividend <= divisor) {
                dividend -= divisor;
                quotient++;
            }
            if (negativeCount == 1) {
                quotient = -quotient;
            }
            return quotient;
        }
    }

    // kind of sort of works
    func divide(_ dividend: Int, _ divisor: Int) -> Int {
        var dividend = Int32(dividend)
        var divisor = Int32(divisor)
        if dividend == Int32.min && divisor == -1 { return Int(Int32.max) }
        var isNegative = false
        if dividend < 0 {
            isNegative = !isNegative
            dividend = -dividend
        }
        if divisor < 0 {
            isNegative = !isNegative
            divisor = -divisor
        }
        
        var quotient = 0
        while divisor - dividend <= 0 {
            quotient += 1
            dividend -= divisor
        }
        return Int(isNegative ? -quotient : quotient)
    }

    public static class Solution2 {
        /**
         * credit: https://leetcode.com/problems/divide-two-integers/solution/ solution 2
         * <p>
         * 1. exponetial growth to check to speed up
         * 2. we still turn all numbers into negatives because negatives are a superset of all numbers in the positives.
         * <p>
         * Time: O(log2n)
         * Space: O(1)
         */
        private static final int HALF_INT_MIN = Integer.MIN_VALUE / 2;

        public int divide(int dividend, int divisor) {
            if (dividend == Integer.MIN_VALUE && divisor == -1) {
                return Integer.MAX_VALUE;
            }
            int negativeCount = 0;
            if (dividend < 0) {
                negativeCount++;
            } else {
                dividend = -dividend;
            }
            if (divisor < 0) {
                negativeCount++;
            } else {
                divisor = -divisor;
            }
            int quotient = 0;
            while (dividend <= divisor) {
                int powerOfTwo = -1;
                int value = divisor;
                while (value >= HALF_INT_MIN && value + value >= dividend) {
                    value += value;
                    powerOfTwo += powerOfTwo;
                }
                quotient += powerOfTwo;
                dividend -= value;
            }
            if (negativeCount != 1) {
                quotient = -quotient;
            }
            return quotient;
        }
    }
}

   // LC33. Search in Rotated Sorted Array
    // https://leetcode.com/problems/search-in-rotated-sorted-array/
    func search(_ nums: [Int], _ target: Int) -> Int { // O(logN)
        if nums.isEmpty { return -1 }
        var left = 0
        var right = nums.count - 1
        while left <= right {
            let mid = left + (right - left) / 2
            if nums[mid] == target { return mid }
            if nums[mid] < nums[right] {
                if nums[mid] < target && target <= nums[right] {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            } else {
                if nums[left] <= target && target < nums[mid] {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            }
        }
        return -1
    }

    // my solution - 7/23/22
    func search(_ nums: [Int], _ target: Int) -> Int {
        var left = 0 
        var right = nums.count - 1
        
        while left <= right {
            let mid = right - (right - left)/2
            if nums[mid] == target { return mid }
            if nums[left] < nums[mid] {
                // rotation is not in this left half
                if target < nums[mid] && target >= nums[left] {
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            } else {
                // rotation is not in the right half
                if target > nums[mid] && target <= nums[right] {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }

        }
        return -1
    }

// Approach 2: One-pass Binary Search
public int search(int[] nums, int target) {
    int start = 0, end = nums.length - 1;
    while (start <= end) {
      int mid = start + (end - start) / 2;
      if (nums[mid] == target) return mid;
      else if (nums[mid] >= nums[start]) {
        if (target >= nums[start] && target < nums[mid]) end = mid - 1;
        else start = mid + 1;
      }
      else {
        if (target <= nums[end] && target > nums[mid]) start = mid + 1;
        else end = mid - 1;
      }
    }
    return -1;
}

    // LC34. Find First and Last Position of Element in Sorted Array
    // Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
    // Your algorithm's runtime complexity must be in the order of O(log n).
    // https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        var result:[Int] = [-1, -1]
        var left = 0
        var right = nums.count - 1
        while left <= right {
            let mid = right - (right - left) / 2
            if nums[mid] == target {
                result[0] = mid
                if mid == 0 || nums[mid - 1] != target { break }                
                right = mid - 1
            } else if nums[mid] > target {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        left = 0
        right = nums.count - 1
        while left <= right {
            let mid = right - (right - left) / 2
            if nums[mid] == target {
                result[1] = mid
                if mid == nums.count - 1 || nums[mid + 1] != target { break }
                left = mid + 1
            } else if nums[mid] < target {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return result
    }

    // LC36. Valid Sudoku
    func isValidHelper(_ board: [[Character]], _ rowStart: Int, _ rowEnd: Int, _ colStart: Int, _ colEnd: Int) -> Bool {
        var set = Set<Character>()
        for rx in rowStart..<rowEnd {
            for cx in colStart..<colEnd {
                if board[rx][cx] == "." { continue }
                if set.contains(board[rx][cx]) {
                    return false
                }
                set.insert(board[rx][cx])
            }
        }
        return true
    }
    
    func isValidSudoku(_ board: [[Character]]) -> Bool {
        
        // check rows
        let rowCount = board.count
        let columnCount = board[0].count
        for rx in 0..<rowCount {
            if !isValidHelper(board, rx, rx + 1, 0, columnCount) {
                return false
            }
        }
        
        // check columns
        for cx in 0..<columnCount {
            if !isValidHelper(board, 0, rowCount, cx, cx + 1) {
                return false
            }
        }        
        
        // check quadrants
        for I in 0..<3 {
            for J in 0..<3 {
                if !isValidHelper(board, I * 3, (I + 1) * 3 , J * 3, (J + 1) * 3) {
                    return false
                }   
            }
        }
        
        return true
    }

    // LC38. Count and Say
    struct Digit {
        init(_ num: Character) {
            self.num = num
        }
        var num:Character = ""
        var count = 1
    }

class Solution {
    func countAndSay(_ n: Int) -> String {
        var result = "1"
        for _ in 1..<n {
            var digits: [Digit] = []
            print(result)
            var ix = 0
            for c in result {
                if ix == 0 {
                    digits.append(Digit(c))
                } else {
                    var lastDigit = digits[digits.count - 1]
                    if lastDigit.num == c {
                        lastDigit.count += 1
                        digits[digits.count - 1] = lastDigit
                    } else {
                        digits.append(Digit(c))
                    }
                }
                ix += 1
            }
            result = ""
            for digit in digits {
                result += "\(digit.count)\(digit.num)"
            }
        }
        return result
    }
}

    // LC46. Permutations
    func permuteHelper(_ px: Int, _ partial: inout [Int], _ result: inout [[Int]]) {
        if px == partial.count {
            result.append(partial)
            return
        }
        for ix in px..<partial.count {
            partial.swapAt(px, ix)
            permuteHelper(px + 1, &partial, &result)
            partial.swapAt(px, ix)
        }
    }
    
    func permute(_ input : [Int]) -> [[Int]]
    {
        var result: [[Int]] = []
        var input = input
        permuteHelper(0, &input, &result)
        return result
    } 

    // LC48. Rotate Image
    func rotate(_ matrix: inout [[Int]]) {
        // transpose
        for rx in 0..<matrix.count - 1 {
            for cx in 0..<matrix[0].count - 1 - rx {
                let mcx = matrix.count - 1 - rx
                let mrx = matrix[0].count - 1 - cx
                let temp = matrix[rx][cx]
                matrix[rx][cx] = matrix[mrx][mcx]
                matrix[mrx][mcx] = temp
            }
        }
        
        // reflect
        for rx in 0..<matrix.count / 2 {
            for cx in 0..<matrix[0].count {
                let mrx = matrix.count - 1 - rx
                let mcx = cx
                let temp = matrix[rx][cx]
                matrix[rx][cx] = matrix[mrx][mcx]
                matrix[mrx][mcx] = temp
            }
        }
    }

    // LC49. Group Anagrams
    func groupAnagrams(_ strs: [String]) -> [[String]] {
        var dict: [String: [String]] = [:]
        for str in strs {
            let key = String(str.sorted())
            dict[key, default: []].append(str)
        }
        return Array(dict.values)
    }

    func myPow(_ x: Double, _ n: Int) -> Double {
        if n == 0 { return 1 }
        if n == 1 {  return x }
        if n < 0 { return 1 / myPow(x, -n) }
        let result = myPow(x, n / 2)
        return n % 2 == 0 ? result * result : x * result * result
    }

    // LC53. Maximum Subarray
    func maxSubArray(_ nums: [Int]) -> Int {
        var maxSum = nums[0]
        var runningSum = nums[0]
        for ix in 1..<nums.count {
            runningSum = max(nums[ix], nums[ix] + runningSum)
            maxSum = max(maxSum, runningSum)
        }
        return maxSum
    }

    // 8/24 run
    func maxSubArray(_ nums: [Int]) -> Int {
        var result = Int.min
        var runningSum = 0
        for ix in 0..<nums.count {
            runningSum += nums[ix]
            result = max(result, runningSum)
            if runningSum < 0 {
                runningSum = 0
            }
        }
        return result
    }

    // LC54. Spiral Matrix
    func spiralOrder(_ matrix: [[Int]]) -> [Int] {
        var rowStart = 0
        var rowEnd = matrix.count - 1
        var colStart = 0
        var colEnd = matrix[0].count - 1
        
        var result: [Int] = []
        while rowStart <= rowEnd && colStart <= colEnd {
            
            for cx in colStart...colEnd {
                result.append(matrix[rowStart][cx])
            }
            
            rowStart += 1
            
            if rowStart > rowEnd { break }
            
            for rx in rowStart...rowEnd {
                result.append(matrix[rx][colEnd])
            }
            
            colEnd -= 1
            
            if colStart > colEnd { break }
            
            for cx in (colStart...colEnd).reversed() {
                result.append(matrix[rowEnd][cx])
            }
            
            rowEnd -= 1
            
            if rowStart > rowEnd { break }
            
            for rx in (rowStart...rowEnd).reversed() {
                result.append(matrix[rx][colStart])
            }
            
            colStart += 1
        }
        return result
    }

    // LC55. Jump Game
    func canJump(_ nums: [Int]) -> Bool { 
        var next = nums[0]
        for ix in 1..<nums.count {
            if next < ix {
                break
            }
            next = max(next, nums[ix] + ix)
        }
        return next >= nums.count - 1
    }


    // LC56. Merge Intervals
    func merge(_ intervals: [[Int]]) -> [[Int]] {
        var result: [[Int]] = []
        guard intervals.count > 0 else { return result }
        let intervals = intervals.sorted { $0[0] < $1[0] }
        var current = intervals[0]
        for ix in 1..<intervals.count {
            if current[1] >= intervals[ix][0] {
                current[0] = min(current[0], intervals[ix][0])
                current[1] = max(current[1], intervals[ix][1])                
            } else {
                result.append(current)
                current = intervals[ix]
            }
        }
        result.append(current)
        return result
    }

    // LC62. Unique Paths
    func uniquePaths(_ m: Int, _ n: Int) -> Int {
        let column = [Int](repeating: 1, count: n)
        var grid: [[Int]] = [[Int]](repeating: column, count: m)
        for rx in 1..<m {
            for cx in 1..<n {
                grid[rx][cx] = grid[rx - 1][cx] + grid[rx][cx - 1]
            }
        }
        return grid[m - 1][n - 1]
    }

    // LC66. Plus One
    func plusOne(_ digits: [Int]) -> [Int] {
        guard digits.count > 0 else { return [] }
        var digits = digits
        var ix = digits.count - 1
        digits[ix] += 1
        while ix > 0 && digits[ix] == 10 {
            digits[ix] = 0
            digits[ix - 1] += 1
            ix -= 1
        }
        if digits[0] == 10 {
            digits[0] = 0
            digits.insert(1, at: 0)
        }
        return digits
    }

    // LC69. Sqrt(x)
    func mySqrt(_ x: Int) -> Int {
       if x == 1 || x == 0 { return x }
        var left = 1,  right = x / 2
        while left <= right {
            let mid = right - (right - left) / 2
            if mid * mid == x { return mid }
            if mid * mid < x { left = mid + 1 }
            else { right = mid - 1 }
        }
        return right
    }

    func mySqrt(_ x: Int) -> Int {
       guard x > 1 else { return x }
       let x = Double(x)
       var x0 = x
       var x1 = (x0 + x / x0) / 2.0
       while (abs(x0 - x1) >= 1) {
           x0 = x1
           x1 = (x0 + x / x0) / 2.0
       } 
       return Int(x1) 
    }

    // LC70. Climbing Stairs
    func climbStairs(_ n: Int) -> Int {
      if n == 1 { return 1 }
      if n == 2 { return 2 }
      var prev = 2
      var prevToPrev = 1
      var result = 0
      for ix in 3...n {
          result = prev + prevToPrev
          prevToPrev = prev
          prev = result
      }  
        return result
    }


    // LC73. Set Matrix Zeroes
    func setZeroes(_ matrix: inout [[Int]]) {
        
        var columnZeroIsZeroes = false
        var rowZeroIsZeroes = false
        
        for ix in 0..<matrix.count {
            for jx in 0..<matrix[0].count {
               if matrix[ix][jx] == 0 {
                   matrix[0][jx] = 0
                   matrix[ix][0] = 0
                   if ix == 0 {
                       rowZeroIsZeroes = true
                   }
                   if jx == 0 {
                       columnZeroIsZeroes = true
                   }
               }
            }
        }
        
        for ix in (1..<matrix.count).reversed() {
            for jx in (1..<matrix[0].count).reversed() { 
                 if matrix[0][jx] == 0 || 
                   matrix[ix][0] == 0 {
                    matrix[ix][jx] = 0
                }
            }
        }
        
        if rowZeroIsZeroes {
            for jx in 0..<matrix[0].count {
                matrix[0][jx] = 0
            }
        }
        if columnZeroIsZeroes {
            for ix in 0..<matrix.count {
                matrix[ix][0] = 0
            }
        }
    }

    func setZeroes(_ matrix: inout [[Int]]) {
        var firstRowZeroes = false
        var firstColumnZeroes = false
        
        for ix in 0..<matrix.count {
            for jx in 0..<matrix[0].count {
                if matrix[ix][jx] == 0 {
                    matrix[ix][0] = 0
                    matrix[0][jx] = 0
                    if ix == 0 {
                        firstRowZeroes = true
                    }
                    if jx == 0 {
                        firstColumnZeroes = true
                    }
                }
            }
        }
        
        for ix in (0..<matrix.count).reversed() {
            for jx in (0..<matrix[0].count).reversed() {
                if (ix != 0 && matrix[ix][0] == 0) || (jx != 0 && matrix[0][jx] == 0) {
                    matrix[ix][jx] = 0
                } else if ix == 0 && firstRowZeroes {
                    matrix[ix][jx] = 0
                } else if jx == 0 && firstColumnZeroes {
                    matrix[ix][jx] = 0
                }
            }
        }
    }

    // LC75. Sort Colors
    func sortColors(_ nums: inout [Int]) {
        guard nums.count > 0 else { return }
        var nextZero = 0
        var nextTwo = nums.count - 1
        var ix = 0
        while ix <= nextTwo {
            if nums[ix] == 0 {
                nums.swapAt(ix, nextZero)
                nextZero += 1
                ix += 1
            } else if nums[ix] == 2 {
                nums.swapAt(ix, nextTwo)
                nextTwo -= 1
            }  else {
                ix += 1
            }
        }
    }

    // LC78. Subsets
    func subsets(_ nums: [Int]) -> [[Int]] {
        let n = nums.count
        var result: [[Int]] = []
        let numSubsets = Int(pow(2.0, Double(n)))
        for ix in 0..<numSubsets {
            var partial: [Int] = []
            for nx in 0..<nums.count {
                if ((1 << nx) & ix) != 0 {
                    partial.append(nums[nx])
                }
            }
            result.append(partial)
        }
        return result
    }

    List<List<Integer>> output = new ArrayList();
    int n, k;

    public void backtrack(int first, ArrayList<Integer> curr, int[] nums) {
        // if the combination is done
        if (curr.size() == k) {
            output.add(new ArrayList(curr));
        return;
        }
        for (int i = first; i < n; ++i) {
            // add i into the current combination
          curr.add(nums[i]);
          // use next integers to complete the combination
          backtrack(i + 1, curr, nums);
          // backtrack
          curr.remove(curr.size() - 1);
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        n = nums.length;
        for (k = 0; k < n + 1; ++k) {
            backtrack(0, new ArrayList<Integer>(), nums);
        }
        return output;
    }

    // LC79. Word Search
    func search(_ board:  inout [[Character]], _ word: String, _ wx: Int, _ ix: Int, _ jx: Int) -> Bool {
        if ix < 0 || ix >= board.count || jx < 0 || jx >= board[0].count {
            return false
        }
        let char = word[word.index(word.startIndex, offsetBy: wx)]
        if char != board[ix][jx] {
            return false
        }
        if wx == word.count - 1 {
            return true
        }
        let shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for shift in shifts {
            board[ix][jx] = " " // to prevent revisiting the same cell
            if search(&board, word, wx + 1, ix + shift[0], jx + shift[1]) == true {
                return true
            }
            board[ix][jx] = char // restore
        }
        return false
    }
    
    func exist(_ board: [[Character]], _ word: String) -> Bool {
        guard word.count > 0 else { return false }
        var board = board
        for ix in 0..<board.count {
            for jx in 0..<board[0].count {
                if search(&board, word, 0, ix, jx) == true {
                    return true
                }
            }
        }
        return false
    }

    // LC88. Merge Sorted Array
    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var n1x = m - 1
        var n2x = n - 1
        var wx = m + n - 1
        while n2x >= 0 {
            if n1x < 0 || nums1[n1x] < nums2[n2x] {
                nums1[wx] = nums2[n2x]
                n2x -= 1
            } else {
                nums1[wx] = nums1[n1x]
                n1x -= 1
            }
            wx -= 1
        }
    }

    // LC94. Binary Tree Inorder Traversal
    func inorderTraversal(_ root: TreeNode?) -> [Int] {
        
        var result: [Int] = []
        guard let root = root else { return result }
        var stack: [TreeNode?] = []

        var curr: TreeNode? = root
        while curr != nil || stack.count > 0 {
            while curr != nil {
                stack.append(curr)
                curr = curr?.left
            }
            
            curr = stack.removeLast()
            if let curr = curr {
                result.append(curr.val)                
            }
            curr = curr?.right
        }
        
        return result
    }
    
    // LC98. Validate Binary Search Tree
    var prev: Int? = nil
    
    func isValidBSTHelper(_ root: TreeNode?) -> Bool {
       guard let root = root else { return true }
       if !isValidBSTHelper(root.left) {
           return false
       }
       if let prev = prev,
            root.val <= prev {
                return false
            }
        prev = root.val
        return isValidBSTHelper(root.right)
    }
    func isValidBST(_ root: TreeNode?) -> Bool {
        return isValidBSTHelper(root)
    }

    // LC101. Symmetric Tree
    func isSymmetric(_ root: TreeNode?) -> Bool {
        guard let root = root else { return true } 
        return isSymmetric(root.left, root.right)
    }

   func isSymmetric(_ left: TreeNode?, _ right: TreeNode?) -> Bool {
      if left == nil && right == nil {
          return true
      } 
      if left?.val != right?.val {
          return false
      }
      return  isSymmetric(left?.left, right?.right) && isSymmetric(left?.right, right?.left)
   }

    // LC102. Binary Tree Level Order Traversal
   func levelOrderHelper(_ root: TreeNode?, _ result: inout [[Int]], _ index: Int) {
        guard let root = root else { return }
        if result.count == index {
            result.append([])
        }
        result[index].append(root.val)
        if let left = root.left {
            levelOrderHelper(left, &result, index + 1)
        } 
        if let right = root.right {
            levelOrderHelper(right, &result, index + 1)
        }
    }
    
    func levelOrder(_ root: TreeNode?) -> [[Int]] {
        guard let root = root else { return [] }
        var result: [[Int]] = []
        levelOrderHelper(root, &result, 0)
        return result
    }

    // LC103. Binary Tree Zigzag Level Order Traversal
    func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
      guard let root = root else { return [] }
      var result = [[Int]]()
      var queue = [root]
      var level = 0
      while !queue.isEmpty {
         let size = queue.count
         var levelResult = [Int]()
         for _ in 0..<size {
            let node = queue.removeFirst()
            levelResult.append(node.val)
            if let left = node.left { queue.append(left) }
            if let right = node.right { queue.append(right) }
         }
         if level % 2 == 1 { levelResult.reverse() }
         result.append(levelResult)
         level += 1
      }
      return result
    }

    // LC104. Maximum Depth of Binary Tree
    func maxDepthHelper(_ root: TreeNode?, _ depth: Int) -> Int {
        guard let root = root else { return depth }
        return max(maxDepthHelper(root.left, depth + 1), maxDepthHelper(root.right, depth + 1))
    }
    
    func maxDepth(_ root: TreeNode?) -> Int {
       return maxDepthHelper(root, 0)
    }

    // LC105. Construct Binary Tree from Preorder and Inorder Traversal
    var inorderMap: [Int:Int] = [:]
    var px = 0
    
    
    func buildTreeHelper(_ preorder: [Int], _ left: Int, _ right: Int) -> TreeNode? {
        if left > right {
            return nil
        }
        let val = preorder[px]
        let root = TreeNode(val)
        px += 1
        if let inorderIx = inorderMap[val] {
            root.left = buildTreeHelper(preorder, left, inorderIx - 1)
            root.right = buildTreeHelper(preorder, inorderIx + 1, right)
        }
        return root
    }
    
    func buildTree(_ preorder: [Int], _ inorder: [Int]) -> TreeNode? {
        for ix in 0..<inorder.count {
            inorderMap[inorder[ix]] = ix
        }
        return buildTreeHelper(preorder, 0, inorder.count - 1)
    }

    // LC108. Convert Sorted Array to Binary Search Tree
    func sortedArrayToBSTHelper(_ nums: [Int], _ left: Int, _ right: Int) -> TreeNode? {
        if left > right {
            return nil
        }   
        let mid = right - (right - left) / 2
        let root = TreeNode(nums[mid])
        root.left = sortedArrayToBSTHelper(nums, left, mid - 1)
        root.right = sortedArrayToBSTHelper(nums, mid + 1, right)
        return root
    }
    
    func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
        return sortedArrayToBSTHelper(nums, 0, nums.count - 1)
    }

    // LC116. Populating Next Right Pointers in Each Node
    func connect(_ root: Node?) -> Node? {
        guard let root = root else { return nil }
        var itr: Node? = root 
        while itr?.left != nil {
            var siblingsItr: Node? = itr
            while siblingsItr != nil {
                siblingsItr?.left?.next = siblingsItr?.right
                siblingsItr?.right?.next = siblingsItr?.next?.left
                siblingsItr = siblingsItr?.next
            }
            itr = itr?.left            
        }
        return root
    }

    // LC118. Pascal's Triangle
    func generate(_ numRows: Int) -> [[Int]] {
      var result = [[Int]]()
      for nx in 0..<numRows {
          var row = [Int](repeating: 1, count: nx + 1)
          var ix = 1
          while ix < row.count - 1 {
            row[ix] = result[result.count - 1][ix - 1] + result[result.count - 1][ix]
            ix += 1
          }
          result.append(row)
      }
      return result
    }

    // LC121. Best Time to Buy and Sell Stock
    func maxProfit(_ prices: [Int]) -> Int {
        var maxProfit = 0
        var minPrice = Int.max
        for price in prices {
            minPrice = min(minPrice, price)
            maxProfit = max(maxProfit, price - minPrice)
        }
        return maxProfit
    }

    // LC125. Valid Palindrome
    func isPalindrome(_ s: String) -> Bool {
        var left = 0
        var right = s.count - 1
        let schars = Array(s)
        while left < right {
            while left < right && !schars[left].isLetter && !schars[left].isNumber {
                left += 1
            }
            while left < right && !schars[right].isLetter && !schars[right].isNumber {
                right -= 1
            }
            if schars[left].lowercased() != schars[right].lowercased() {
                return false
            }
            left += 1
            right -= 1
        }
        return true
    }

    // LC128. Longest Consecutive Sequence
    func longestConsecutive(_ nums: [Int]) -> Int {
        var set = Set<Int>(nums)
        
        var maxLen = 0
        for num in nums {
            if set.remove(num) != nil {
                var count = 1
                var val = num
                while set.remove(val - 1) != nil {
                    val -= 1
                }
                count += (num - val)
                val = num
                while set.remove(val + 1) != nil {
                    val += 1
                }
                count += (val - num)
                maxLen = max(maxLen, count)
            }
        }     
        return maxLen
    }

    // LC136. Single Number
    func singleNumber(_ nums: [Int]) -> Int {
        var result = nums[0]
        for ix in 1..<nums.count {
            result ^= nums[ix]
        }
        return result
    }

    // LC138. Copy List with Random Pointer
    var visitedNodes: [Node:Node] = [:] 
    
    func copyRandomList(_ head: Node?) -> Node? {
        
        guard let head = head else { return nil }
        
        if let stored = visitedNodes[head] {
            return stored
        }
        
        let newNode = Node(head.val)
        visitedNodes[head] = newNode        
        newNode.next = copyRandomList(head.next)
        newNode.random = copyRandomList(head.random)
        
        return newNode
    }

    // LC141. Linked List Cycle
    func hasCycle(_ head: ListNode?) -> Bool {
        guard head != nil else { return false }
        var slow = head
        var fast = head?.next
        while slow !== fast {
            if fast == nil || fast?.next == nil { return false }
            fast = fast?.next?.next
            slow = slow?.next
        }
        return true
    }

    func hasCycle(_ head: ListNode?) -> Bool {
        guard head != nil else { return false }
        var fast: ListNode? = head
        var slow: ListNode? = head
        fast = fast?.next?.next
        slow = slow?.next
        while fast !== nil && slow !== nil && fast !== slow {
            fast = fast?.next?.next
            slow = slow?.next
        }
        return fast !== nil && fast === slow
    }

    // LC150. Evaluate Reverse Polish Notation
    func evalRPN(_ tokens: [String]) -> Int {
        var stack: [Int] = []
        let functionLookup: [String: (Int, Int) -> Int] = [
            "+": { $0 + $1 },
            "-": { $0 - $1 },
            "*": { $0 * $1 },
            "/": { $0 / $1 }
        ]
        for token in tokens {
            if let function = functionLookup[token] {
                let second = stack.removeLast()
                let first = stack.removeLast()
                stack.append(function(first, second))
            } else {
                stack.append(Int(token) ?? 0)
            }           
        }
        return stack.removeLast()
    }

    // LC152. Maximum Product Subarray
    func maxProduct(_ nums: [Int]) -> Int {
        guard nums.count > 0 else { return Int.min }
        var max_so_far = nums[0]
        var min_so_far = nums[0]
        var result = nums[0]
        for ix in 1..<nums.count {
            let num = nums[ix]
            let temp_max = max(num, max(max_so_far * num, min_so_far * num))
            min_so_far = min(num, min(max_so_far * num, min_so_far * num))
            max_so_far = temp_max
            result = max(result, max_so_far)
        }
        return result
    }

// LC155. Min Stack
class MinStack {

    var stack: [Int] = []
    var minStack: [Int] = []
    
    init() {
        
    }
    
    func push(_ val: Int) {
        stack.append(val)
        if minStack.count == 0 || minStack[minStack.count - 1] >= val {
            minStack.append(val)
        }
    }
    
    func pop() {
        let removedVal = stack.removeLast()
        if minStack[minStack.count - 1] == removedVal {
            minStack.removeLast()
        }
    }
    
    func top() -> Int {
        return stack[stack.count - 1]
    }
    
    func getMin() -> Int {
        return minStack[minStack.count - 1]
    }
}

    // LC160. Intersection of Two Linked Lists
    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
        // one pointer goes a + c + b and other pointer goes b + c + a
        var pA = headA
        var pB = headB
        while pA !== pB {
            pA = pA == nil ? headB : pA?.next
            pB = pB == nil ? headA : pB?.next            
        }
        return pA
    }


    // LC162. Find Peak Element
    func findPeakElement(_ nums: [Int]) -> Int {
        var left = 0
        var right = nums.count - 1
        while left < right {
            let mid = left - (left - right)/2
            if nums[mid] > nums[mid + 1] {
                right = mid
            } else {
                left = mid + 1
            }
        }
        return left
    }

    // LC163. Missing Ranges
    func findMissingRanges(_ nums: [Int], _ lower: Int, _ upper: Int) -> [String] {
        var result: [String] = []
        var prev = lower - 1
        for ix in 0...nums.count {
            var curr = ix < nums.count ? nums[ix] : upper + 1
            if prev + 1 <= curr - 1 {
                result.append(missingRangeString(prev + 1, curr - 1))
            }
            prev = curr
        }
        return result
    }
    
    func missingRangeString(_ lower: Int, _ upper: Int) -> String {
        if lower == upper {
            return "\(lower)"
        } else {
            return "\(lower)->\(upper)"
        }
    }

    // LC169. Majority Element
    func majorityElement(_ nums: [Int]) -> Int {
        var votes = 1
        var candidate = nums[0]
        for ix in 1..<nums.count {
            if votes == 0 {
                candidate = nums[ix]
            }                        
            votes += nums[ix] == candidate ? 1 : -1
        }
        return candidate
    }

    // LC171. Excel Sheet Column Number
    func titleToNumber(_ columnTitle: String) -> Int {
        var result = 0
        let map = ["A": 1, "B": 2]
        let s = columnTitle
        let a: Character = "A"
        for ix in 0..<s.count {
            let c = columnTitle[s.index(s.startIndex, offsetBy: ix)]
            let mappedVal = (c.asciiValue ?? 0) - (a.asciiValue ?? 0) + 1
            result = result * 26 + Int(mappedVal) ?? 0
        }
        return result
    }

    // LC179. Largest Number
    func largestNumber(_ nums: [Int]) -> String {
        let nums = nums.map { String($0) }   
        let numsSorted = nums.sorted {
            $0 + $1 > $1 + $0
        }
        let str = numsSorted.reduce("", +)
        if str.first == "0" {
            return "0"
        }
        return str
    }
    // LC189. Rotate Array
    func reverse(_ nums: inout [Int], _ left: Int, _ right: Int) {
        var left = left
        var right = right
        while left < right {
            nums.swapAt(left, right)
            left += 1
            right -= 1
        }
    }
    
    func rotate(_ nums: inout [Int], _ k: Int) {
        let k = k % nums.count
        reverse(&nums, 0, nums.count - k - 1)
        reverse(&nums, nums.count - k, nums.count - 1)
        reverse(&nums, 0, nums.count - 1)
    }

    // LC190. Reverse Bits
    uint32_t reverseBits(uint32_t n) {
         n = (n >> 16) | (n << 16);
        n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8);
        n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4);
        n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2);
        n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1);
        return n;
    }

    // LC191. Number of 1 Bits
    func hammingWeight(_ n: Int) -> Int {
        var n = n
        var count = 0
        while n > 0 {
            count += 1
            n = n & (n - 1)
        }
        return count
    }

    // LC200. Number of Islands
    func explore(_ grid: inout [[Character]], _ ix: Int, _ jx: Int) {
        if ix < 0 || ix >= grid.count || jx < 0 || jx >= grid[0].count || grid[ix][jx] != "1" {
            return
        }
        let shifts = [[0, -1], [0, 1], [1, 0], [-1, 0]]
        grid[ix][jx] = "X"
        for shift in shifts {
            let nextIx = ix + shift[0]
            let nextJx = jx + shift[1]
            explore(&grid, nextIx, nextJx)
        }
    }
    
    func numIslands(_ grid: [[Character]]) -> Int {
        var grid = grid
        var count = 0
        for ix in 0..<grid.count {
            for jx in 0..<grid[0].count {
                if grid[ix][jx] == "1" {
                    count += 1
                    explore(&grid, ix, jx)
                }
            }
        }
        return count
    }

    // LC202. Happy Number
    func getNextN(_ n : Int) -> Int {
        var totalSum = 0
        var n = n
        while n > 0 {
            let d = n % 10
            n = n / 10
            totalSum += (d * d)
        }
        return totalSum
    }
    
    func isHappy(_ n: Int) -> Bool {
        var slow = n
        var fast = getNextN(n)
        
        while fast != 1 && slow != fast {
            slow = getNextN(slow)
            fast = getNextN(getNextN(fast))
        }
        return fast == 1
    }

    // LC204. Count Primes
    func countPrimes(_ n: Int) -> Int {
        guard n > 1 else { return 0 }
        var sieved = [Bool](repeating: false, count: n + 1)
        var count = 0

        for ix in 2..<n {
            if sieved[ix] == false {
                sieved[ix] = true
                count += 1
                var jx = 2
                while (ix * jx < n) {
                    sieved[ix * jx] = true
                    jx += 1
                }
            }   
        }
        return count
    }

    // LC206. Reverse Linked List
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

    // LC208. Implement Trie (Prefix Tree)
    class Trie {

    var terminal = false
    var root: [Character: Trie] = [:]
    
    init() {
        
    }
    
    func insert(_ word: String) {
        if word.count == 0 { 
            terminal = true
            return 
        }
        let suffix = String(word.suffix(from: word.index(word.startIndex, offsetBy: 1)))
        if let first = word.first {
            let trie = root[first, default: Trie()]
            trie.insert(suffix)            
            root[first] = trie
        }
    }
    
    func search(_ word: String) -> Bool {
        if word.count == 0 && terminal == true {
            return true
        }
        if let first = word.first, let storedTrie = root[first] {
            let suffix = String(word.suffix(from: word.index(word.startIndex, offsetBy: 1)))    
            return storedTrie.search(suffix)
        }
        return false
    }
    
    func startsWith(_ prefix: String) -> Bool {
        if prefix.count == 0 {
            return true
        }
        if let first = prefix.first, let storedTrie = root[first] {
            let suffix = String(prefix.suffix(from: prefix.index(prefix.startIndex, offsetBy: 1)))    
            return storedTrie.startsWith(suffix)
        }
        return false
        
    }
}

        public int findKthLargest(int[] nums, int k) {
            Arrays.sort(nums);
            return nums[nums.length - k];
        }

        public int findKthLargest(int[] nums, int k) {
            PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
            for (int i : nums) {
                maxHeap.offer(i);
            }
            while (k-- > 1) {
                maxHeap.poll();
            }
            return maxHeap.poll();
        }

    // LC217. Contains Duplicate
    func containsDuplicate(_ nums: [Int]) -> Bool {
        let set = Set(nums)
        return set.count != nums.count    
    }

    // LC230. Kth Smallest Element in a BST
    var nodeCount = 0
    
    func kthSmallestHelper(_ root: TreeNode?, _ k: Int) -> TreeNode? {
        guard let root = root else { return nil }
        if let left = root.left {
            let leftNode = kthSmallestHelper(root.left, k)
            if leftNode != nil { return leftNode }
        }
        nodeCount += 1        
        if (nodeCount == k) {
            return root
        }
        if let right = root.right {
            let rightNode = kthSmallestHelper(root.right, k)
            if rightNode != nil { return rightNode }
        }
        return nil
    }
    
    func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
        guard let root = root else { return Int.max }
        let node = kthSmallestHelper(root, k)
        return node?.val ?? Int.max
    }

    // LC234. Palindrome Linked List
    public boolean isPalindrome(ListNode head) {

        if (head == null) return true;

        // Find the end of first half and reverse second half.
        ListNode firstHalfEnd = endOfFirstHalf(head);
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);

        // Check whether or not there is a palindrome.
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) result = false;
            p1 = p1.next;
            p2 = p2.next;
        }        

        // Restore the list and return the result.
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }

    // Taken from https://leetcode.com/problems/reverse-linked-list/solution/
    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode nextTemp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nextTemp;
        }
        return prev;
    }

    private ListNode endOfFirstHalf(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    // LC236. Lowest Common Ancestor of a Binary Tree
    var answer: TreeNode? = nil
    
    func lowestCommonAncestorHelper(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> Bool {
        guard let root = root else { return false }
        let left = lowestCommonAncestorHelper(root.left, p, q) ? 1 : 0
        let right = lowestCommonAncestorHelper(root.right, p, q) ? 1 : 0
        let mid = (p === root || q === root) ? 1 : 0
        
        if left + right + mid >= 2 { 
            answer = root
            return true 
        }
        
        return left + right + mid > 0
    }
    
    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        lowestCommonAncestorHelper(root, p, q);
        return answer
    }

    // LC237. Delete Node in a Linked List
    func deleteNode(_ node: ListNode?) {
        node?.val = node?.next?.val ?? 0
        node?.next = node?.next?.next
    }

    // LC238. Product of Array Except Self
    func productExceptSelf(_ nums: [Int]) -> [Int] {
        var answer = [Int](repeating: 1, count: nums.count)
        
        for ix in 1..<nums.count {
            answer[ix] = answer[ix - 1] * nums[ix - 1]
        }
        
        var R = 1
        for jx in (0...nums.count - 1).reversed() {
            answer[jx] = answer[jx] * R
            R *= nums[jx]
        }
        
        return answer
    }

    func productExceptSelf(_ nums: [Int]) -> [Int] {
        var result = [Int](repeating: 1, count: nums.count)
        for ix in 1..<nums.count {
            result[ix] = result[ix - 1] * nums[ix - 1]
        }
        var R = nums[nums.count - 1] // could crash for empty array
        for ix in (0..<nums.count - 1).reversed() {
            result[ix] = result[ix] * R
            R *= nums[ix]
        }
        return result
    }

    // LC240. Search a 2D Matrix II
    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
        
        if matrix.count == 0 { return false }
        
        var col = matrix[0].count - 1
        var row = 0
        
        while row < matrix.count && col >= 0 {
            if target > matrix[row][col] {
                row += 1
            } else if target < matrix[row][col] {
                col -= 1
            } else {
                return true
            }
        }
        return false
    }

    // LC242. Valid Anagram
    func isAnagram(_ s: String, _ t: String) -> Bool {
        guard s.count == t.count else { return false }
        var buckets: [Int] = [Int](repeating: 0, count: 26)
        
        let refAscii = Int(Character("a").asciiValue ?? 0) 
        
        for ix in 0..<s.count {
            let schar = s[s.index(s.startIndex, offsetBy: ix)]
            var asciiValue = Int(schar.asciiValue ?? 0) - refAscii
            buckets[asciiValue] += 1
            
            let tchar = t[t.index(t.startIndex, offsetBy: ix)]            
            asciiValue = Int(tchar.asciiValue ?? 0) - refAscii
            buckets[asciiValue] -= 1
        }
        
        for bucket in buckets {
            if bucket != 0 {
                return false
            }
        }
        
        return true
    }

// LC251. Flatten 2D Vector
class Vector2D {

    var outer = 0
    var inner = 0
    var vector: [[Int]] = []
    
    init(_ vec: [[Int]]) {
        vector = vec
    }
    
    func next() -> Int {
        advanceToNext()
        let nextValue = vector[outer][inner]
        inner += 1
        return nextValue
    }
  
    func advanceToNext() {
        while outer < vector.count && inner == vector[outer].count {
            inner = 0
            outer += 1
        }
    }
    
    func hasNext() -> Bool {
        advanceToNext()        
        return outer < vector.count
    }
}


    // LC268. Missing Number
    func missingNumber(_ nums: [Int]) -> Int {
        let sum = nums.reduce(0, +)
        let expectedSum = nums.count * (nums.count + 1) / 2
        return expectedSum - sum
    }

    // LC277. Find the Celebrity
    func findCelebrity(_ n: Int) -> Int {
        var candidate = 0
        for ix in 0..<n {
            if knows(candidate, ix) {
                candidate = ix
            }
        }
        for ix in 0..<n {
            if ix != candidate && knows(candidate, ix) || !knows(ix, candidate) {
                return -1
            }
        }
        return candidate
    }

    // LC283. Move Zeroes
    func moveZeroes(_ nums: inout [Int]) {
        var wx = 0
        for rx in 0..<nums.count {
            if nums[rx] != 0 {
                nums.swapAt(wx, rx)
                wx += 1
            }
        }
    }

    // LC285. Inorder Successor in BST
    func inorderSuccessor(_ root: TreeNode?, _ p: TreeNode?) -> TreeNode? {
        var successor: TreeNode? = nil
        var itr = root
        while itr != nil {
            if let p = p {
                if let itrVal = itr?.val {
                    if p.val >= itrVal {
                        itr = itr?.right
                    } else {
                        successor = itr
                        itr = itr?.left
                    }
                }
            }
        }
        return successor
    }

    // LC287. Find the Duplicate Number
    public int findDuplicate(int[] nums) {
        
        // Find the intersection point of the two runners.
        int tortoise = nums[0];
        int hare = nums[0];
        
        do {
            tortoise = nums[tortoise];
            hare = nums[nums[hare]];
        } while (tortoise != hare);

        // Find the "entrance" to the cycle.
        tortoise = nums[0];
        
        while (tortoise != hare) {
            tortoise = nums[tortoise];
            hare = nums[hare];
        }

        return hare;
    }

    // LC324. Wiggle Sort II
        public void wiggleSort(int[] nums) {
            int median = findKthLargest(nums, (nums.length + 1) / 2);
            int n = nums.length;

            int left = 0;
            int i = 0;
            int right = n - 1;

            while (i <= right) {

                if (nums[newIndex(i, n)] > median) {
                    swap(nums, newIndex(left++, n), newIndex(i++, n));
                } else if (nums[newIndex(i, n)] < median) {
                    swap(nums, newIndex(right--, n), newIndex(i, n));
                } else {
                    i++;
                }
            }
        }

        private int findKthLargest(int[] nums, int k) {
            PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
            for (int i : nums) {
                maxHeap.offer(i);
            }
            while (k-- > 1) {
                maxHeap.poll();
            }
            return maxHeap.poll();
        }

        private void swap(int[] nums, int i, int j) {
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
        }

        private int newIndex(int index, int n) {
            return (1 + 2 * index) % (n | 1);
        }

    // LC326. Power of Three
    func isPowerOfThree(_ n: Int) -> Bool {
        //  return (Math.log10(n) / Math.log10(3)) % 1 == 0;
        // return (Math.log(n) / Math.log(3) + epsilon) % 1 <= 2 * epsilon;
        return n > 0 && 1162261467 % n == 0;
    }

    // LC328. Odd Even Linked List
    public ListNode oddEvenList(ListNode head) {
        if (head == null) return null;
        ListNode odd = head, even = head.next, evenHead = even;
        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    // LC340. Longest Substring with At Most K Distinct Characters
    func lengthOfLongestSubstringKDistinct(_ s: String, _ k: Int) -> Int {
        var maxLen = 0
        var left = 0
        var right = 0
        var map: [Character: Int] = [:]
        let schars = Array(s)
        while right < schars.count {
            map[schars[right]] = right
            if map.count == k + 1 {
                if let minIndex = map.values.min() {
                    map[schars[minIndex]] = nil
                    left = minIndex + 1                    
                }
            }
            maxLen = max(maxLen, right - left + 1)
            right += 1                        
        }
        return maxLen
    }

// LC341. Flatten Nested List Iterator
class NestedIterator {

    var stack: [NestedInteger]!
    
    init(_ nestedList: [NestedInteger]) {
        if nestedList.count == 0 {
            return
        }
        stack = nestedList.reversed()
    }
    
    func advanceToNext() {
        while stack.count > 0 && !stack[stack.count - 1].isInteger() {
            let top = stack.removeLast()
            stack.append(contentsOf: top.getList().reversed())
        }
    }
    
    func next() -> Int {
        stack.removeLast().getInteger()
    }
    
    func hasNext() -> Bool {
        advanceToNext()
        return stack.count > 0
    }
}


    // LC344. Reverse String
    func reverseString(_ s: inout [Character]) {
        var i = 0
        var j = s.count - 1
        while i < j {
            s.swapAt(i, j) 
            i += 1
            j -= 1
        }
    }

    // LC347. Top K Frequent Elements
    public int[] topKFrequent(int[] nums, int k) {
        // O(1) time
        if (k == nums.length) {
            return nums;
        }
        
        // 1. build hash map : character and how often it appears
        // O(N) time
        Map<Integer, Integer> count = new HashMap();
        for (int n: nums) {
          count.put(n, count.getOrDefault(n, 0) + 1);
        }

        // init heap 'the less frequent element first'
        Queue<Integer> heap = new PriorityQueue<>(
            (n1, n2) -> count.get(n1) - count.get(n2));

        // 2. keep k top frequent elements in the heap
        // O(N log k) < O(N log N) time
        for (int n: count.keySet()) {
          heap.add(n);
          if (heap.size() > k) heap.poll();    
        }

        // 3. build an output array
        // O(k log k) time
        int[] top = new int[k];
        for(int i = k - 1; i >= 0; --i) {
            top[i] = heap.poll();
        }
        return top;
    }
    
    int[] unique;
    Map<Integer, Integer> count;

    public void swap(int a, int b) {
        int tmp = unique[a];
        unique[a] = unique[b];
        unique[b] = tmp;
    }

    public int partition(int left, int right, int pivot_index) {
        int pivot_frequency = count.get(unique[pivot_index]);
        // 1. move pivot to end
        swap(pivot_index, right);
        int store_index = left;

        // 2. move all less frequent elements to the left
        for (int i = left; i <= right; i++) {
            if (count.get(unique[i]) < pivot_frequency) {
                swap(store_index, i);
                store_index++;
            }
        }

        // 3. move pivot to its final place
        swap(store_index, right);

        return store_index;
    }
    
    public void quickselect(int left, int right, int k_smallest) {
        /*
        Sort a list within left..right till kth less frequent element
        takes its place. 
        */

        // base case: the list contains only one element
        if (left == right) return;
        
        // select a random pivot_index
        Random random_num = new Random();
        int pivot_index = left + random_num.nextInt(right - left); 

        // find the pivot position in a sorted list
        pivot_index = partition(left, right, pivot_index);

        // if the pivot is in its final sorted position
        if (k_smallest == pivot_index) {
            return;    
        } else if (k_smallest < pivot_index) {
            // go left
            quickselect(left, pivot_index - 1, k_smallest);     
        } else {
            // go right 
            quickselect(pivot_index + 1, right, k_smallest);  
        }
    }
    
    public int[] topKFrequent(int[] nums, int k) {
        // build hash map : character and how often it appears
        count = new HashMap();
        for (int num: nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }
        
        // array of unique elements
        int n = count.size();
        unique = new int[n]; 
        int i = 0;
        for (int num: count.keySet()) {
            unique[i] = num;
            i++;
        }
        
        // kth top frequent element is (n - k)th less frequent.
        // Do a partial sort: from less frequent to the most frequent, till
        // (n - k)th less frequent element takes its place (n - k) in a sorted array. 
        // All element on the left are less frequent.
        // All the elements on the right are more frequent. 
        quickselect(0, n - 1, n - k);
        // Return top k frequent elements
        return Arrays.copyOfRange(unique, n - k, n);
    }

// LC348. Design Tic-Tac-Toe
class TicTacToe {
    var board: [[Int]]!
    var n: Int!
    
    init(_ n: Int) {
        self.n = n
        let row = [Int](repeating: 0, count: n)
        board = [[Int]](repeating: row, count: n)
    }
    
    private func checkRow(_ player: Int, _ row: Int) -> Bool {
        for jx in 0..<n {
            if board[row][jx] != player {
                return false
            }
        }
        return true
    }
    
    private func checkColumn(_ player: Int, _ col: Int) -> Bool {
        for rx in 0..<n {
            if board[rx][col] != player {
                return false
            }
        }    
        return true
    }
    
    private func checkDiagonal(_ player: Int, _ row: Int, _ col: Int) -> Bool {
        for ix in 0..<n {
            if board[ix][ix] != player {
                return false
            } 
        }
        return true
    }
    
    private func checkAntiDiag(_ player: Int, _ row: Int, _ col: Int) -> Bool {
        for rx in 0..<n {
            let cx = n - 1 - rx
            if board[rx][cx] != player {
                return false
            }
        }
        return true
    }
    
    private func isWinner(_ player: Int, _ row: Int, _ col: Int) -> Bool {
        if checkRow(player, row) == true ||
            checkColumn(player, col) == true ||
            row == col && checkDiagonal(player, row, col) ||
            col == n - 1 - row && checkAntiDiag(player, row, col)  {
                return true
        }
        return false
    }
    
    func move(_ row: Int, _ col: Int, _ player: Int) -> Int {
        board[row][col] = player
        if isWinner(1, row, col) {
            return 1
        } else if isWinner(2, row, col) {
            return 2
        } else {
            return 0
        }
    }
}

public class TicTacToe {
    int[] rows;
    int[] cols;
    int diagonal;
    int antiDiagonal;

    public TicTacToe(int n) {
        rows = new int[n];
        cols = new int[n];
    }

    public int move(int row, int col, int player) {
        int currentPlayer = (player == 1) ? 1 : -1;
        // update currentPlayer in rows and cols arrays
        rows[row] += currentPlayer;
        cols[col] += currentPlayer;
        // update diagonal
        if (row == col) {
            diagonal += currentPlayer;
        }
        //update anti diagonal
        if (col == (cols.length - row - 1)) {
            antiDiagonal += currentPlayer;
        }
        int n = rows.length;
        // check if the current player wins
        if (Math.abs(rows[row]) == n ||
                Math.abs(cols[col]) == n ||
                Math.abs(diagonal) == n ||
                Math.abs(antiDiagonal) == n) {
            return player;
        }
        // No one wins
        return 0;
    }
}

    // LC350. Intersection of Two Arrays II
    func intersect(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        if nums1.count > nums2.count {
        return intersect(nums2, nums1)
    }
    var map = [Int: Int]()
    for num in nums1 {
        map[num, default: 0] += 1
    }

    var result = [Int]()
    for num in nums2 {
        if let value = map[num] {
            if value > 0 {
                result.append(num)
                map[num, default: 0] -= 1
            }
        }
    }
    return result
    }

    // LC371. Sum of Two Integers
    func getSum(_ a: Int, _ b: Int) -> Int {
        var a = a
        var b = b
         while b != 0 {
            let answer = a ^ b
            let carry = (a & b) << 1
            a = answer
            b = carry
        }
        
        return a
    }

    // LC384. Shuffle an Array
    var original: [Int] = []
    var shuffled: [Int] = []
    
    init(_ nums: [Int]) {
        original = nums
        shuffled = nums
    }
    
    func reset() -> [Int] {
        shuffled = original
        return original
    }
    
    func shuffle() -> [Int] {
        for ix in 0..<shuffled.count {
            shuffled.swapAt(ix, Int.random(in: ix..<shuffled.count))
        }
        return shuffled
    }

    // LC387. First Unique Character in a String
    func firstUniqChar(_ s: String) -> Int {
        
       var buckets: [Int] = [Int](repeating: 0, count: 26)
       for char in s {
           let index = (char.asciiValue ?? 0) - (Character("a").asciiValue ?? 0)
           buckets[Int(index)] += 1
       }
        
       var ix = 0
       for char in s {
            let index = (char.asciiValue ?? 0) - (Character("a").asciiValue ?? 0)
           if buckets[Int(index)] == 1 {
               return ix
           }
           ix += 1
       } 
       return -1
    }

    // LC412. Fizz Buzz
    func fizzBuzz(_ n: Int) -> [String] {
        var result:[String] = [String](repeating: "", count: n)
        
        for ix in 1...n {
            if ix % 3 == 0 &&  ix % 5 == 0 {
                result[ix - 1] = "FizzBuzz"
            } else if ix % 3 == 0 {
                result[ix - 1] = "Fizz"
            } else if ix % 5 == 0 {
                result[ix - 1] = "Buzz"
            } else {
                result[ix - 1] = String(ix)
            }
        }
        return result
    }

    // LC454. 4Sum II
    func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
        var result = 0
        var map = [Int: Int]()
    for num1 in nums1 {
        for num2 in nums2 {
            let sum = num1 + num2
            map[sum, default: 0] += 1
        }
    }
    for num3 in nums3 {
        for num4 in nums4 {
            let sum = num3 + num4
            result += map[-sum, default: 0]
        }
    }
    return result
    }



