export def sync-uv [] {
  print "Syncing dependencies with uv..."
  try {
    uv sync

    if $env.LAST_EXIT_CODE == 0 {
      print $"(ansi green_bold)Environment is ready!(ansi reset)"
    } else {
      error make {msg: "uv sync finished with errors. Check the output above."}
    }
  } catch {
    print $"(ansi red_bold)Fatal: uv command failed to execute.(ansi reset)"
  }
}

export def run-with-cuda [script: string] {
  print "Checking CUDA availability via uv..."

  let cuda_check = (
    do { uv run python -c "import torch; print(torch.cuda.is_available())" } | complete  | get stdout | str trim
  )

  if $cuda_check == "True" {
    print $"(ansi green_bold)CUDA detected. Running ($script)...(ansi reset)"
    uv  run python $script
  } else {
    print $"(ansi yellow_bold)CUDA not found!(ansi reset)"
    let ans = (input "Do you want to run on CPU anyway? (y/n): ")

    if $ans == "y" {
      "Running on CPU..."
      uv run python $script
    } else {
      print "Aborted."
    }
  }
}

export def test-cpu-fallback [script: string] {
  print "Forcing No-CUDA environment..."
  with-env { CUDA_VISIBLE_DEVICES: "-1" } {
    run-with-cuda $script
  }
}
