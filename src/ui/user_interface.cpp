
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

#include <string>

#include <imgui.h>
#include <imgui-SFML.h>
#include <portable-file-dialogs.h>

#include "user_interface.hpp"

using namespace std::literals::string_literals;

namespace usdg
{
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
	  if(_video_player)
	    _video_player.reset();

	  auto result = pfd::open_file(
	    "Select File"s, "../data",
	    { "Image Files", "*.png *.jpg *.jpeg *.bmp *.tga *.gif *.psd *.hdr *.pic"
	     // "Video Files"s, "*.mp4 *.wav", automate this by ffmpeg -demuxers
	    }).result();

	  if(!result.empty())
	  {
	    _video_player.emplace(_opt_manager.best(), result[0]);
	  }
	}
	ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Action"))
      {
	// if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
	// if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {}  // Disabled item
	// ImGui::Separator();
	// if (ImGui::MenuItem("Cut", "CTRL+X")) {}
	// if (ImGui::MenuItem("Copy", "CTRL+C")) {}
	// if (ImGui::MenuItem("Paste", "CTRL+V")) {}
	
	ImGui::EndMenu();
      }
      ImGui::EndMainMenuBar();
    }
  }

  void
  UserInterface::
  state_render()
  {
    auto state = _state;

    this->render_menubar();
    _linesearch.render();
    if(_video_player)
    {
      _video_player->render();
    }
  }

  void
  UserInterface::
  state_action()
  {
    switch(_state)
    {
    case UIState::idle:
      if (_linesearch.is_select_pressed())
      {
	_linesearch.enable_select_button();
      }
      break;

    case UIState::rendering:
    {
      double beta = _linesearch.selected_parameter();
      if (_linesearch.is_select_pressed())
      {
	_opt_manager.find_next_query(beta);
      }

      auto param  = _opt_manager.query(beta);
      _video_player->update_parameter(param);
      break;
    }

    case UIState::optimized:
      _linesearch.enable_select_button();
      [[fallthrougth]];
    case UIState::optimizing:
    {
      double beta = _linesearch.selected_parameter();
      auto param  = _opt_manager.query(beta);
      _video_player->update_parameter(param);
      break;
    }
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
      if (_linesearch.is_select_pressed())
	_state = UIState::optimizing;
      else
	_state = UIState::rendering;
      break;

    case UIState::optimizing:
      if (_opt_manager.is_optimizing())
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
