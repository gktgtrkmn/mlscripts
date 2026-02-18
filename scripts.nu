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
