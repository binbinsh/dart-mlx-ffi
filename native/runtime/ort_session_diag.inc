void Session::Diagnostics(
    DinfDiagBuilder* out,
    const std::string& prefix) const {
  out->AddString(dinf_diag_path(prefix, "engine"), "onnx");
  out->AddString(dinf_diag_path(prefix, "provider"), provider_);
  out->AddBool(dinf_diag_path(prefix, "provider_appended"), provider_appended_);
  out->AddInt(dinf_diag_path(prefix, "num_threads"), num_threads_);
  out->AddBool(dinf_diag_path(prefix, "io_binding"), use_io_binding_);
  out->AddBool(dinf_diag_path(prefix, "device_outputs"), use_device_outputs_);
  out->AddBool(dinf_diag_path(prefix, "output_views"), use_output_views_);
  out->AddBool(dinf_diag_path(prefix, "sync_bound_inputs"), sync_bound_inputs_);
  out->AddBool(dinf_diag_path(prefix, "sync_bound_outputs"), sync_bound_outputs_);
  out->AddBool(dinf_diag_path(prefix, "cache_bound_outputs"), cache_bound_outputs_);
  if (cuda_graph_id_ >= 0) {
    out->AddInt(dinf_diag_path(prefix, "cuda_graph_id"), cuda_graph_id_);
  }
  out->AddStringList(
      dinf_diag_path(prefix, "cpu_output_names"),
      cpu_output_names_);
  out->AddStringList(
      dinf_diag_path(prefix, "device_output_names"),
      device_output_names_);
  if (!prepacked_weights_key_.empty()) {
    out->AddString(
        dinf_diag_path(prefix, "prepacked_weights_key"),
        prepacked_weights_key_);
  }
  out->AddStringList(
      dinf_diag_path(prefix, "available_providers"),
      available_providers_);
  out->AddStringList(dinf_diag_path(prefix, "input_names"), input_names_);
  out->AddStringList(dinf_diag_path(prefix, "output_names"), output_names_);
  AddTensorMetadataList(
      out,
      dinf_diag_path(prefix, "input_metadata"),
      input_metadata_);
  AddTensorMetadataList(
      out,
      dinf_diag_path(prefix, "output_metadata"),
      output_metadata_);
}
