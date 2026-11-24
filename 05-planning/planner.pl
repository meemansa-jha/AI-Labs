% Planning lab - Tasks 8-10: Prolog as an independent logical verifier.
% Run:  swipl -q -s planner.pl -g run_all -t halt

% ---- Task 8: warehouse connectivity facts ---------------------------------
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

% ---- Task 9: checking moves proposed by the Python planner -----------------
valid_move(X,Y) :-
    connected(X,Y).

% A whole plan of moves is valid if every consecutive move is valid.
% Example:  ?- valid_plan(a, [b,c]).   (Move(a,b), Move(b,c))
valid_plan(_, []).
valid_plan(From, [To|Rest]) :-
    valid_move(From, To),
    valid_plan(To, Rest).

% ---- Task 10: facts + rules -> inference -----------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% ---- Driver that prints every query result ---------------------------------
show(Goal) :-
    ( call(Goal) -> R = true ; R = false ),
    format("?- ~w.~t~30|~w~n", [Goal, R]).

run_all :-
    writeln('Task 8'),
    show(can_move(a,b)),
    show(can_move(a,c)),
    writeln('Task 9'),
    show(valid_move(a,b)),
    show(valid_move(b,c)),
    show(valid_move(a,c)),
    show(valid_plan(a,[b,c])),
    show(valid_plan(a,[c])),
    writeln('Task 10'),
    show(reduce_speed).
