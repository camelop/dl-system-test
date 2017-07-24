# 要求支持的接口

## 默认:

1. Session
  1. Session().run(**[node], feed-dict, isGlobalInit, name**)
  2. \__enter__
  3. \__exit__
2. float32
3. zeros
4. random_normal
5. constant


## 运算符:

1. Variable
2. placeholder
3. \__add__
4. \__mul__
5. \__radd__
6. \__rmul__
7. reduce_sum
8. assign
9. gradients
10. equal
11. argmax
12. log
13. reduce_mean
14. nn.softmax
15. global_variables_initializer
16. train.GradientDescentOptimizer().minimize()
17. cast
18. matmul
19. nn.relu
20. nn.softmax_cross_entropy_with_logits
21. train.AdamOptimizer().minimize
22. nn.conv2d
23. nn.max_pool
24. reshape
25. nn.dropout
