open System
open System.IO

open Accord
open Accord.MachineLearning
open Accord.Math

// http://www.codesuji.com/2017/08/27/K-Means-Clustering-with-F/

let config = File.ReadAllLines("application.config") 
             |> Array.map(fun x -> x.Split(':') |> Array.map (fun y -> y.Trim()))  
             |> Array.map(fun x -> (x.[0], x.[1]))
             |> Map.ofArray

// https://gist.github.com/paralax/0801babfba22bfa1a0316863330c6667    
let ngrams (s : string) (n: int) : Map<string,int> = 
    s.ToCharArray() 
    |> Array.map string
    |> Seq.windowed n  
    |> Seq.map (String.concat "")
    |> Seq.groupBy (fun x -> x) 
    |> Seq.map (fun (x,y)->  x, Seq.length y) 
    |> Map.ofSeq

//let Q (i:string) (q : Map<string,int>) : double =
let prob (i:string) (q:Map<string,int>) : double =
    // retrieves the frequency of i in q if found or returns .00001
    let qq = Map.toList q |> List.map snd |> List.sum |> float
    match (Map.tryFind i q) with
    | Some(x) -> float(x)/qq
    | None -> 0.00001
    
// https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
let KullbackLeiblerD (p : Map<string,int>) (q : Map<string,int> ) : float[] = 
    (Map.toList p |> List.map fst) @ (Map.toList q |> List.map fst) 
    |> Set.ofList 
    |> Set.toSeq
    |> Seq.map (fun x -> (x, prob x p))
    |> Seq.map (fun (x,y) -> y * System.Math.Log(y/(prob x q), 2.0))
    |> Seq.map float
    |> Seq.toArray

// http://accord-framework.net/docs/html/T_Accord_MachineLearning_KMeans.htm
let clusters(n: int) (samples: Map<string,int>[]) : string [] =
    let km = new KMeans(n)
    km.Distance = KullbackLeiblerD |> ignore
    let clusters = km.Learn(samples)
    clusters.Decide(samples)

let binaryStrings(path: string) : string [] =
    let p = new Diagnostics.Process()
    p.StartInfo.FileName <- "/usr/bin/strings"
    p.StartInfo.Arguments <- String.Format("{0}", path)
    p.StartInfo.RedirectStandardOutput <- true
    p.StartInfo.UseShellExecute <- false
    p.Start() |> ignore
    p.StandardOutput.ReadToEnd().Split('\n')

let todaysHaul(path: string) : Map<string,int>[] =
    Directory.GetFiles(config.["kippodldir"], DateTime.Today.ToString("yyyyMMdd*"))
    |> Array.map binaryStrings
    |> Array.map (fun x -> ngrams (String.Concat('\n', x)) 1)

type CommandLineOptions = {
    n: int        // -n N
}

let defaultOptions = {
    n = 5
}

let usage args =
    printfn "classify_samples.exe ARGS"
    printfn "arguments and options:"
    printfn "  -n N    Number of KMeans clusters to build (default:%d)" defaultOptions.n
    printfn "  -h      this text"

// inspired via https://fsharpforfunandprofit.com/posts/pattern-matching-command-line/
let rec parseCommandLine args soFar : CommandLineOptions = 
    match args with
    | [] -> soFar    
    | "-n"::xs ->
        let t = int (List.head xs)
        let rem = List.tail xs
        parseCommandLine rem { soFar with n=t}
    | "-h"::xs -> 
        usage args
        exit 0
    | x::xs ->
        printfn "WARNING option %s is not understood" x
        parseCommandLine xs soFar

[<EntryPoint>]
let main args = 
    let options = parseCommandLine (List.ofArray args) defaultOptions
    todaysHaul
    |> clusters (options.n)
    0