# Regex expression based text transformer

```
How to implement the relative unknown word reposition transformer with regex module?


Did you see AA? => I see AA.
Did you see BB? => I see BB.

A deep learning framework needs to be able to handle new data more than AA or BB, for example CC, and generate the right response "I see *" based on general rule "Did you see *".

But if you use regex, you could also make it.

All you have to do is do auto regex expression generation from pure text data.

For example, when you face "multiple key link to one value" case, you merge those keys into one general regex expression.


#ai #idea #yingshaoxo
```

```
We will use char level operation to get unknown keywords regex from "multiple key -> one value" data pairs

Hi AA -> Hi you.
Hi BB -> Hi you.
Hi CC -> Hi you.

We need to get "Hi (.*?) -> Hi you." from above data automatically.



Did you see AA? => I see AA.
Did you see BB? => I see BB.

We need to get "Did you see (?P<someone>.*?)? -> I see {someone}." from above data automatically.



That is steven, my uncle. => I see, steven is your uncle.
That is wind_god, my uncle. => I see, wind_god is your uncle.

We need to get "That is (?P<name>.*?), my uncle. => I see, {name} is your uncle." from above data automatically.
```
