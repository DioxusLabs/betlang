module StringMap = Map.Make (String)

let normalize value =
  value |> String.trim |> String.lowercase_ascii

let count values =
  List.fold_left
    (fun table value ->
      let key = normalize value in
      let previous = Option.value (StringMap.find_opt key table) ~default:0 in
      StringMap.add key (previous + 1) table)
    StringMap.empty values

let () =
  count [ "Rust"; "Rust"; "Python" ]
  |> StringMap.iter (fun key value -> Printf.printf "%s=%d\n" key value)
