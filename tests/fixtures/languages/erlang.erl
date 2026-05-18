-module(betlang_fixture).
-export([greet/1, count/1]).

greet(Name) when is_list(Name) ->
    io:format("hello ~s~n", [Name]).

count(Values) ->
    count(Values, #{}).

count([], Acc) ->
    Acc;
count([Head | Tail], Acc) ->
    Current = maps:get(Head, Acc, 0),
    count(Tail, maps:put(Head, Current + 1, Acc)).
