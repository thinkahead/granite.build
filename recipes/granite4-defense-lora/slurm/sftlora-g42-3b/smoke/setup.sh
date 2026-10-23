set -eu
sed -i s/fcntl.flock/fcntl.lockf/ /usr/local/lib/python3.12/dist-packages/filelock/_unix.py
perl -0777 -i -pe 's/\n {12}with suppress\(OSError\):\n {16}Path\(self\.lock_file\)\.unlink\(\)//' /usr/local/lib/python3.12/dist-packages/filelock/_unix.py
echo "FILELOCK_PATCH_APPLIED lockf=$(grep -c fcntl.lockf /usr/local/lib/python3.12/dist-packages/filelock/_unix.py) unlink=$(grep -c 'unlink()' /usr/local/lib/python3.12/dist-packages/filelock/_unix.py)"
grep -q "fcntl.lockf" /usr/local/lib/python3.12/dist-packages/filelock/_unix.py
test "$(grep -c 'unlink()' /usr/local/lib/python3.12/dist-packages/filelock/_unix.py)" = 2
sed -i "s|self.log_file_path =.*$|self.log_file_path = '/tmp/odm.jsonl'|g" /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py
sed -i 's/vocab_size=32000/vocab_size=100352/g' /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py
test "$(grep -c 'vocab_size=100352' /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py)" = 2
perl -0777 -i -pe 's{        return self\._update_sampling_ratio\(self\.sampling_weights\)}{        self.sampling_weights = self.sampling_weights / self.sampling_weights.max()\n        return self._update_sampling_ratio(self.sampling_weights)}' /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py
perl -0777 -i -pe 's{self\.eta \* est_r / self\.total_categories}{min(self.eta * est_r / self.total_categories, 30.0)}' /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py
grep -q "sampling_weights.max()" /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py
grep -q "min(self.eta" /usr/local/lib/python3.12/dist-packages/fms_acceleration_odm/odm/dataset.py
