for id in $(seq 1319023 1319500); do
  STATUS=$(bjobs -o "jobid stat exit_code" -noheader $id 2>/dev/null)
  if echo "$STATUS" | grep -q "EXIT.*1"; then
    echo "[ERROR] $STATUS"
  else
    echo "[OK] $STATUS"
  fi
done
