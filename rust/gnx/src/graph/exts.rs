// Graph-specific PartialEq, Eq, and Hash traits
// Unlike their std counterparts, these consider
// also the underlying DAG structure e.g.
// if a = b = c = d
//    (&a, &b) == (&c, &d)
//    (&a, &a) != (&c, &d)
