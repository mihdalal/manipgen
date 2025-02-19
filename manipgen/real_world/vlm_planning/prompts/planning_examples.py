""" 
### Plan examples ###
Here are some examples of input tasks and the corresponding plans that you should generate and the type of format we are looking for.
MAKE SURE YOU USE THE SAME FORMAT AS THE EXAMPLES BELOW FOR YOUR OUTPUT PLANS.
Example 1:
Task: Put away all remaining items on the table in the shelf.
Output:
[
    ("spoon", "pick"),
    ("shelf", "place"),
    ("turquoise pot", "pick"),
    ("shelf", "place"),
    ("plastic bottle", "pick"),
    ("shelf", "place"),
]

Example 2:
Task: Clean up the table by putting everything that is not in the bin into the bin.
Output:
[
    ("yellow plastic bottle", "pick"),
    ("bin", "place"),
    ("spoon", "pick"),
    ("bin", "place"),
    ("turquoise pot", "pick"),
    ("bin", "place"),
    ("blue plastic bottle", "pick"),
    ("bin", "place"),
    ("white toy kettle", "pick"),
    ("bin", "place"),
    ("white roll of tape", "pick"),
    ("bin", "place"),
]
"""