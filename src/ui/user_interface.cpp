
/*
 * Copyright (C) 2021  Ray Kim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "user_interface.hpp"

#include "natural_sort.hpp"

#include <imgui.h>
#include <imgui-SFML.h>
#include <portable-file-dialogs.h>

#include <string>
#include <fstream>
#include <filesystem>
#include <ranges>

using namespace std::literals::string_literals;
namespace fs = std::filesystem;

namespace usdg
{
  template <typename WidgetType>
  void
  toggle_view(std::optional<WidgetType>& widget)
  {
    if(widget)
      widget.reset();
    else
      widget.emplace();
  }

  std::pair<UserInterface::FilePaths,
	    UserInterface::FilePaths>
  UserInterface::
  process_data_directory(std::string const& path)
  {
    auto files_iterator = fs::directory_iterator(path) ;
    auto files          = std::vector<fs::directory_entry>(
      fs::begin(files_iterator),
      fs::end(files_iterator));

    auto pfm_fnames_view =  std::ranges::ref_view(files)
      | std::ranges::views::filter(
	[](auto const& fentry){
	  return fentry.path().extension() == ".pfm";
	})
      | std::ranges::views::transform([](auto const& fentry){
	return std::string(fentry.path());
      });
    auto pfm_fnames = std::vector<std::string>(
      pfm_fnames_view.begin(), pfm_fnames_view.end());
    SI::natural::sort(pfm_fnames.begin(), pfm_fnames.end());

    auto mask_fname_view = std::ranges::ref_view(files)
      | std::ranges::views::filter(
	[](auto const& fentry){
	  return fentry.path().extension() == ".png";
	})
      | std::ranges::views::transform([](auto const& fentry){
	return std::string(fentry.path());
      });
    auto mask_fnames = std::vector<std::string>(mask_fname_view.begin(),
						mask_fname_view.end());
    return {pfm_fnames, mask_fnames};
  }

  void
  UserInterface::
  render_menubar()
  {
    if (ImGui::BeginMainMenuBar())
    {
      if (ImGui::BeginMenu("File"))
      {
	if (ImGui::MenuItem("Open File"))
	{
	  auto selection = pfd::select_folder("Select Data Directory"s, "../data").result();

	  if (!selection.empty())
	  { 
	    auto [pfm_fnames, mask_fnames] = process_data_directory(selection);
	    if (pfm_fnames.size() == 0 || mask_fnames.size() < 1)
	    {
	      pfd::message("Data Directory Error",
			   "The selected directory does not contain envelope (*.pfm) files and a single mask (*.png) file",
			   pfd::choice::ok);
	    }
	    else
	    {
	      _opt_manager.emplace();
	      _linesearch.emplace();
	      _video_player.emplace(_opt_manager->best(),
				    pfm_fnames,
				    *mask_fnames.begin());
	    }
	  }
	}

	if (ImGui::MenuItem("Save History"))
	{
	  auto serialized = _opt_manager->serialize();
	  auto fname      = pfd::save_file("Select Save File", "history.json").result();
	  if (!fname.empty())
	  {
	    auto stream     = std::ofstream(fname);
	    stream << serialized;
	    stream.close();
	  }
	}

	if (ImGui::MenuItem("Load History"))
	{
	  if (_opt_manager && _linesearch)
	  {
	    auto fname      = pfd::open_file("Select History File"s, ".").result();
	    if (!fname.empty())
	    {
	      auto stream     = std::ifstream(fname[0]);
	      auto serialized = std::string(std::istreambuf_iterator<char>(stream),
					    std::istreambuf_iterator<char>());
	      stream.close();
	      _opt_manager->deserialize(std::move(serialized));
	      _linesearch->disable_select_button();
	      _state = UIState::optimizing;

	      size_t iter = _opt_manager->iteration();
	      _linesearch->update_iteration(iter);
	    }
	  }
	}

	if (ImGui::MenuItem("Load Presets"))
	{
	  auto fname  = pfd::open_file("Select Preset File"s, ".").result();
	  if (!fname.empty())
	  {
	    _opt_manager.emplace();
	    auto stream = std::ifstream(fname[0]);
	    auto serialized = std::string(std::istreambuf_iterator<char>(stream),
					  std::istreambuf_iterator<char>());
	    stream.close();
	    _opt_manager->load_preset(std::move(serialized));
	  }
	}

	if (ImGui::MenuItem("Export"))
	{
	  auto selection = pfd::select_folder("Select Export Path"s, "../data").result();
	  if (!selection.empty() && _video_player)
	  { 
	    _video_player->export_files(selection, _opt_manager->best());
	  }
	}
	ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("View"))
      {
	// if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
	// if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
	// ImGui::Separator();
	// if (ImGui::MenuItem("Cut", "CTRL+X")) {}
	// if (ImGui::MenuItem("Copy", "CTRL+C")) {}
	// if (ImGui::MenuItem("Paste", "CTRL+V")) {}

	if (ImGui::MenuItem("Parameter Value"))
	{
	  usdg::toggle_view(_param_value_view);
	}
	ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Action"))
      {
	if (ImGui::MenuItem("Reset"))
	{
	  if(_state != UIState::optimizing
	     && _opt_manager)
	  {
	    _opt_manager.emplace();
	    _linesearch.emplace();
	  }
	}
	ImGui::EndMenu();
      }

      ImGui::EndMainMenuBar();
    }
  }

  void
  UserInterface::
  state_render()
  {
    this->render_menubar();
    if(_linesearch)
    {
      _linesearch->render();
    }
    if (_video_player)
    {
      _video_player->render();
    }
    if (_param_value_view)
    {
      _param_value_view->render();
    }
  }

  void
  UserInterface::
  state_action()
  {
    switch(_state)
    {
    case UIState::idle:
      if (_linesearch && _linesearch->is_select_pressed())
      {
	_linesearch->enable_select_button();
      }
      break;

    case UIState::rendering:
      if (_linesearch && _opt_manager)
      {
	double beta = _linesearch->selected_parameter();
	if (_linesearch->is_select_pressed())
	{
	  _opt_manager->find_next_query(beta);
	}

	auto param  = _opt_manager->query(beta);
	if (_video_player)
	  _video_player->update_parameter(param);
	if(_param_value_view)
	  _param_value_view->update_parameter(param);
      }
      break;

    case UIState::optimized:
      if (_linesearch && _opt_manager)
      {
	_linesearch->enable_select_button();
	size_t iter = _opt_manager->iteration();
	_linesearch->update_iteration(iter);
      }
      if (_video_player && _opt_manager)
      {
	_video_player->update_preview(_opt_manager->best());
      }
      [[fallthrough]];
    case UIState::optimizing:
      if (_linesearch && _opt_manager)
      {
	double beta = _linesearch->selected_parameter();
	auto param  = _opt_manager->query(beta);
	if (_video_player)
	  _video_player->update_parameter(param);
	if (_param_value_view)
	  _param_value_view->update_parameter(param);
      }
      break;
    }
  }

  void
  UserInterface::
  state_transition()
  {
    switch(_state)
    {
    case UIState::idle:
      if (_video_player)
	_state = UIState::rendering;
      else
	_state = UIState::idle;
      break;

    case UIState::rendering:
      if (_linesearch && _linesearch->is_select_pressed())
	_state = UIState::optimizing;
      else
	_state = UIState::rendering;
      break;

    case UIState::optimizing:
      if (_opt_manager && _opt_manager->is_optimizing())
	_state = UIState::optimizing;
      else
	_state = UIState::optimized;
      break;

    case UIState::optimized:
      _state = UIState::rendering;
      break;
    }
  }

  std::ostream&
  operator<<(std::ostream& os,
	     usdg::UserInterface const& ui)
  {
    switch(ui._state)
    {
    case UserInterface::UIState::idle:
      os << "current state: idle";
      break;

    case UserInterface::UIState::rendering:
      os << "current state: rendering";
      break;
      
    case UserInterface::UIState::optimizing:
      os << "current state: optimizing";
      break;

    case UserInterface::UIState::optimized:
      os << "current state: optimizing";
      break;
    }
    return os;
  }
}
